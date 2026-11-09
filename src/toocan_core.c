/*
 * toocan_core.c  --  exact, allocation-free port of the TOOCAN 3-D segmentation core
 *
 * Reference (original) implementation being reproduced BIT-FOR-BIT:
 *   toocan-python/toocan/lib/detection_spreading/label.c      Label_region()
 *   toocan-python/toocan/lib/detection_spreading/spreading.py spread_labels_fast()  (numba)
 *   toocan-python/toocan/lib/detection_spreading/detect_and_spread.py  (threshold loop)
 *
 * What is changed relative to the original (software engineering only, no science):
 *   1. Label_region allocated+initialised NBMAX_LABEL_OBJECTS=1e6 Blob structs, each with
 *      a 1000-int neighbour list and a ZSIZE-int per-frame array (~4.3 GB of memset, 2e6
 *      calloc/free) on EVERY threshold iteration (24x per chunk).  Here Blob entries are
 *      created lazily; untouched entries are provably no-ops in every original loop.
 *   2. The cold-pixel index list (np.where + np.sort) is built by an ordered, OpenMP
 *      parallel two-pass scan (ascending linear index preserved).
 *   3. spread: the original queues EVERY labelled voxel (raster order) and then runs a FIFO
 *      BFS.  A labelled voxel without an eligible unlabelled neighbour does nothing and can
 *      never become eligible later (labels only grow, mask/temperature are static), so only
 *      "frontier" voxels are queued -- in the same raster order.  FIFO order, and therefore
 *      the tie-breaking between competing labels, is unchanged.
 *   4. np.unique(list_clusters) printing, array copies, uint8 masks: removed (no effect on labels).
 *
 * All floating point arithmetic mirrors the original operation by operation (see comments
 * "EXACT:"), compiled WITHOUT -ffast-math and with FMA contraction disabled.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define NBMAX_LABEL_OBJECTS      1000000
#define NBMAX_NEIGHBORED_OBJECTS 1000

typedef struct {
    int T, Y, X;           /* ZSIZE, YSIZE, XSIZE of this chunk                         */
    int minBT, maxBT, stepBT;
    int delta_spread;      /* deltaBT_Spread (int in the original)                      */
    int final_delta;       /* 10 in the original (hard coded in detect_and_spread)      */
    int timin;             /* minAreaSeed  (km2 per frame)                              */
    int lifemin;           /* minLifetime  (frames)                                     */
    int nbMaxCluster;      /* size of the persistent table p[] in the original          */
    int quant_mode;        /* 0: (int16)((double)bt*100)  1: (int16)(float(bt*100f))    */
    int fast;              /* 0: scan every cold voxel (literal port)  1: restricted scan (exact, see below) */
} tc_params;

typedef struct {
    double t_quant, t_spread1, t_spread2, t_spread3;
    double t_coldlist, t_scan, t_merge, t_accum, t_filter, t_remap;
    long long n_cold_total, n_label_calls, n_spread_queue_total, n_spread_claims_total;
    long long n_raw_labels_total, n_new_mcs_total;
    double t_spread_q, t_spread_bfs, t_bins, t_ulist, t_vset;
    long long n_U0_total, n_V_total, n_asg_total, n_spread_skipped;
    int err;
} tc_stats;

/* ------------------------------------------------------------------ utilities */
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef struct { int64_t *v; size_t n, cap; } vec64;
static void v64_push(vec64 *a, int64_t x) {
    if (a->n == a->cap) { a->cap = a->cap ? a->cap * 2 : 1024; a->v = (int64_t*)realloc(a->v, a->cap * sizeof(int64_t)); }
    a->v[a->n++] = x;
}

/* ------------------------------------------------------------------ Blob entry */
typedef struct {
    int label;
    int seed_area;            /* int, accumulated as int = int + float (EXACT)      */
    int slotBegin, slotEnd;
    int flagFIX;
    int seed_duration;
    int nb_neighbours;
    int flagRelabel;
    int labelMCS_alreadyidentified;
    int *voisin;              /* neighbour label list, capacity <= 1000             */
    int vcap;
    int *perframe;            /* seed_area_perFrame[ZSIZE]                          */
} Ent;

typedef struct {
    Ent *e; long n, cap;
    int T;
} EntTab;

static int ent_ensure(EntTab *t, long idx) {
    if (idx < 0 || idx >= NBMAX_LABEL_OBJECTS) return -1;
    if (idx >= t->cap) {
        long nc = t->cap ? t->cap : 1024;
        while (nc <= idx) nc *= 2;
        t->e = (Ent*)realloc(t->e, nc * sizeof(Ent));
        memset(t->e + t->cap, 0, (nc - t->cap) * sizeof(Ent));
        for (long i = t->cap; i < nc; i++) { t->e[i].slotBegin = t->T; t->e[i].slotEnd = 0; }
        t->cap = nc;
    }
    if (idx >= t->n) t->n = idx + 1;
    return 0;
}
static void ent_free_all(EntTab *t) {
    for (long i = 0; i < t->cap; i++) { free(t->e[i].voisin); free(t->e[i].perframe); }
    free(t->e); t->e = NULL; t->n = t->cap = 0;
}

/* persistent table p[]: labels of the "already identified MCS" (flagFIX==1). */
typedef struct {
    int *label;               /* p[idx].label, idx = label-1-labelMin               */
    long cap;
} tc_state;

tc_state *tc_state_new(void) { return (tc_state*)calloc(1, sizeof(tc_state)); }
void tc_state_free(tc_state *s) { if (s) { free(s->label); free(s); } }

static int pfix_set(tc_state *s, long idx, int lab) {
    if (idx < 0) return -1;
    if (idx >= s->cap) {
        long nc = s->cap ? s->cap : 4096;
        while (nc <= idx) nc *= 2;
        s->label = (int*)realloc(s->label, nc * sizeof(int));
        memset(s->label + s->cap, 0, (nc - s->cap) * sizeof(int));
        s->cap = nc;
    }
    s->label[idx] = lab;
    return 0;
}

/* Equivalent of cluster_tools.clean_clusters(): keep only the labels present in the overlap. */
int tc_begin_chunk(tc_state *s, const int32_t *labels_present, long n, int labelMin) {
    if (s->label) memset(s->label, 0, s->cap * sizeof(int));
    for (long k = 0; k < n; k++)
        if (pfix_set(s, (long)labels_present[k] - 1 - labelMin, labels_present[k])) return -1;
    return 0;
}

/* ------------------------------------------------------------------ cold list */
/* all linear indices i (ascending) with bt[i] > 0 && bt[i] <= thr   (NaN -> false) */
static void build_cold_list(const float *bt, int64_t nvox, float thr, vec64 *out, int nth) {
    out->n = 0;
#ifdef _OPENMP
    if (nth > 1) {
    int64_t *cnt = (int64_t*)calloc(nth + 1, sizeof(int64_t));
    int64_t blk = (nvox + nth - 1) / nth;
    #pragma omp parallel num_threads(nth)
    {
        int t = omp_get_thread_num();
        int64_t a = t * blk, b = a + blk; if (b > nvox) b = nvox;
        int64_t c = 0;
        for (int64_t i = a; i < b; i++) { float v = bt[i]; c += (v > 0.0f && v <= thr); }
        cnt[t + 1] = c;
    }
    for (int t = 0; t < nth; t++) cnt[t + 1] += cnt[t];
    if ((size_t)cnt[nth] > out->cap) {
        out->cap = (size_t)cnt[nth] + 1024; out->v = (int64_t*)realloc(out->v, out->cap * sizeof(int64_t));
    }
    #pragma omp parallel num_threads(nth)
    {
        int t = omp_get_thread_num();
        int64_t a = t * blk, b = a + blk; if (b > nvox) b = nvox;
        int64_t w = cnt[t];
        for (int64_t i = a; i < b; i++) { float v = bt[i]; if (v > 0.0f && v <= thr) out->v[w++] = i; }
    }
    out->n = (size_t)cnt[nth];
    free(cnt);
    return;
    }
#endif
    for (int64_t i = 0; i < nvox; i++) { float b = bt[i]; if (b > 0.0f && b <= thr) v64_push(out, i); }
}


/* ------------------------------------------------------------------ sorting / list helpers (fast mode) */
static void sort_i64(int64_t *a, size_t n, int bits) {
    if (n < 2) return;
    if (n < 64) {                                   /* insertion sort */
        for (size_t i = 1; i < n; i++) { int64_t v = a[i]; size_t j = i; while (j > 0 && a[j-1] > v) { a[j] = a[j-1]; j--; } a[j] = v; }
        return;
    }
    int64_t *tmp = (int64_t*)malloc(n * sizeof(int64_t));
    int64_t *src = a, *dst = tmp;
    const int D = 11, R = 1 << D;
    size_t *cnt = (size_t*)malloc(R * sizeof(size_t));
    for (int shift = 0; shift < bits; shift += D) {
        memset(cnt, 0, R * sizeof(size_t));
        for (size_t i = 0; i < n; i++) cnt[(src[i] >> shift) & (R - 1)]++;
        size_t sum = 0; for (int r = 0; r < R; r++) { size_t c = cnt[r]; cnt[r] = sum; sum += c; }
        for (size_t i = 0; i < n; i++) dst[cnt[(src[i] >> shift) & (R - 1)]++] = src[i];
        int64_t *t = src; src = dst; dst = t;
    }
    if (src != a) memcpy(a, src, n * sizeof(int64_t));
    free(tmp); free(cnt);
}
static int bits_for(int64_t n) { int b = 1; while (((int64_t)1 << b) < n) b++; return b; }

/* Sorted-unique via a bitmap over the whole volume (parallel set + ordered parallel extraction).
   a[0..n) -> a[0..m) ascending without duplicates, returns m.  bm must hold nvox/64+1 zeroed words and is
   returned zeroed.  Much cheaper than a radix sort when n is a sizeable fraction of nvox/64. */
static size_t sort_unique_bitmap(int64_t *a, size_t n, int64_t nvox, uint64_t *bm, int nth) {
    const size_t nw = (size_t)(nvox / 64 + 1);
    if (nth < 1) nth = 1;
#ifdef _OPENMP
    if (nth > 1 && n > 20000) {
        #pragma omp parallel for num_threads(nth) schedule(static)
        for (size_t i = 0; i < n; i++) __atomic_fetch_or(&bm[a[i] >> 6], (uint64_t)1 << (a[i] & 63), __ATOMIC_RELAXED);
        size_t *cnt = (size_t*)calloc(nth + 1, sizeof(size_t));
        size_t blk = (nw + nth - 1) / nth;
        #pragma omp parallel num_threads(nth)
        {
            int t = omp_get_thread_num(); size_t w0 = (size_t)t * blk, w1 = w0 + blk; if (w1 > nw) w1 = nw;
            size_t c = 0; for (size_t w = w0; w < w1; w++) c += (size_t)__builtin_popcountll(bm[w]);
            cnt[t + 1] = c;
        }
        for (int t = 0; t < nth; t++) cnt[t + 1] += cnt[t];
        #pragma omp parallel num_threads(nth)
        {
            int t = omp_get_thread_num(); size_t w0 = (size_t)t * blk, w1 = w0 + blk; if (w1 > nw) w1 = nw;
            size_t o = cnt[t];
            for (size_t w = w0; w < w1; w++) {
                uint64_t x = bm[w];
                while (x) { int b = __builtin_ctzll(x); a[o++] = (int64_t)(w * 64 + b); x &= x - 1; }
                bm[w] = 0;
            }
        }
        size_t m = cnt[nth]; free(cnt); return m;
    }
#endif
    for (size_t i = 0; i < n; i++) bm[a[i] >> 6] |= (uint64_t)1 << (a[i] & 63);
    size_t o = 0;
    for (size_t w = 0; w < nw; w++) { uint64_t x = bm[w]; if (!x) continue;
        while (x) { int b = __builtin_ctzll(x); a[o++] = (int64_t)(w * 64 + b); x &= x - 1; } bm[w] = 0; }
    return o;
}
/* scratch for sort_list (one chunk at a time per process: the library is NOT re-entrant) */
static uint64_t *g_bm = NULL; static int g_nth = 1;
/* choose: bitmap for large lists, radix otherwise.  Returns the new length (duplicates removed when bitmap used). */
static size_t sort_list(int64_t *a, size_t n, int64_t nvox, uint64_t *bm, int nth) {
    if (bm && n > 50000) return sort_unique_bitmap(a, n, nvox, bm, nth);
    sort_i64(a, n, bits_for(nvox));
    return n;
}


/* out <- in[k] for which lab[in[k]] == 0, order preserved */
static void pack_unlabelled(const int64_t *in, size_t n, const int32_t *lab, vec64 *out, int nth) {
    out->n = 0;
    if (n > out->cap) { out->cap = n + 1024; out->v = (int64_t*)realloc(out->v, out->cap * sizeof(int64_t)); }
#ifdef _OPENMP
    if (nth > 1 && n > 100000) {
        size_t *cnt = (size_t*)calloc(nth + 1, sizeof(size_t));
        size_t blk = (n + nth - 1) / nth;
        #pragma omp parallel num_threads(nth)
        {
            int t = omp_get_thread_num(); size_t a = t * blk, b = a + blk; if (b > n) b = n;
            size_t c = 0; for (size_t i = a; i < b; i++) c += (lab[in[i]] == 0);
            cnt[t + 1] = c;
        }
        for (int t = 0; t < nth; t++) cnt[t + 1] += cnt[t];
        #pragma omp parallel num_threads(nth)
        {
            int t = omp_get_thread_num(); size_t a = t * blk, b = a + blk; if (b > n) b = n;
            size_t w = cnt[t]; for (size_t i = a; i < b; i++) if (lab[in[i]] == 0) out->v[w++] = in[i];
        }
        out->n = cnt[nth]; free(cnt); return;
    }
#endif
    for (size_t i = 0; i < n; i++) if (lab[in[i]] == 0) out->v[out->n++] = in[i];
}

/* merge of two sorted, disjoint lists */
static void merge_sorted(const int64_t *a, size_t na, const int64_t *b, size_t nb, vec64 *out) {
    size_t n = na + nb;
    if (n > out->cap) { out->cap = n + 1024; out->v = (int64_t*)realloc(out->v, out->cap * sizeof(int64_t)); }
    size_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) out->v[k++] = (a[i] < b[j]) ? a[i++] : b[j++];
    while (i < na) out->v[k++] = a[i++];
    while (j < nb) out->v[k++] = b[j++];
    out->n = n;
}

/* merge of two sorted lists, removing duplicates (also inside each list) */
static void merge_unique(const int64_t *a, size_t na, const int64_t *b, size_t nb, vec64 *out, vec64 *scratch) {
    size_t n = na + nb;
    if (n > scratch->cap) { scratch->cap = n + 1024; scratch->v = (int64_t*)realloc(scratch->v, scratch->cap * sizeof(int64_t)); }
    size_t i = 0, j = 0, k = 0;
    while (i < na || j < nb) {
        int64_t x;
        if (j >= nb || (i < na && a[i] <= b[j])) x = a[i++]; else x = b[j++];
        if (k == 0 || scratch->v[k - 1] != x) scratch->v[k++] = x;
    }
    scratch->n = k;
    /* swap result into out */
    int64_t *tv = out->v; size_t tc = out->cap;
    out->v = scratch->v; out->cap = scratch->cap; out->n = k;
    scratch->v = tv; scratch->cap = tc; scratch->n = 0;
}

/* per-chunk binning of the cold voxels by the first threshold at which they are cold (ascending index inside a bin) */
typedef struct { int K; int thr0, step; int64_t *off; int64_t *idx; } tc_bins;
static inline int bin_of(float b, const tc_params *P, int K) {
    if (!(b > 0.0f)) return -1;
    int k;
    if (b <= (float)P->minBT) k = 0;
    else { k = (int)((b - (float)P->minBT) / (float)P->stepBT); if (k < 0) k = 0; if (k > K) k = K; }
    while (k > 0 && b <= (float)(P->minBT + (k - 1) * P->stepBT)) k--;
    while (k < K && !(b <= (float)(P->minBT + k * P->stepBT))) k++;
    return k < K ? k : -1;
}
static void build_bins(const tc_params *P, const float *bt, int64_t nvox, tc_bins *B, int nth) {
    int K = 0; for (int thr = P->minBT; thr < P->maxBT + P->stepBT; thr += P->stepBT) K++;
    B->K = K;
    int64_t *cnt = (int64_t*)calloc((size_t)(nth + 1) * (K + 1), sizeof(int64_t));  /* [t][k] */
    int64_t blk = (nvox + nth - 1) / nth;
    #pragma omp parallel num_threads(nth)
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif
        int64_t a = t * blk, b = a + blk; if (b > nvox) b = nvox;
        int64_t *c = cnt + (size_t)t * (K + 1);
        for (int64_t i = a; i < b; i++) { int k = bin_of(bt[i], P, K); if (k >= 0) c[k]++; }
    }
    B->off = (int64_t*)calloc(K + 1, sizeof(int64_t));
    int64_t total = 0;
    int64_t *start = (int64_t*)calloc((size_t)(nth + 1) * (K + 1), sizeof(int64_t));
    for (int k = 0; k < K; k++) {
        B->off[k] = total;
        for (int t = 0; t < nth; t++) { start[(size_t)t * (K + 1) + k] = total; total += cnt[(size_t)t * (K + 1) + k]; }
    }
    B->off[K] = total;
    B->idx = (int64_t*)malloc((size_t)(total > 0 ? total : 1) * sizeof(int64_t));
    #pragma omp parallel num_threads(nth)
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif
        int64_t a = t * blk, b = a + blk; if (b > nvox) b = nvox;
        int64_t *w = start + (size_t)t * (K + 1);
        for (int64_t i = a; i < b; i++) { int k = bin_of(bt[i], P, K); if (k >= 0) B->idx[w[k]++] = i; }
    }
    free(cnt); free(start);
}

/* V = { cold voxels v (interior x,y) whose 11-point operator contains an element of U0 } , ascending */
static void build_V(const tc_params *P, const float *bt, float thr, const int64_t *U0, size_t nU0,
                    uint8_t *vm, vec64 *V)
{
    const int T = P->T, Ys = P->Y, Xs = P->X; const int64_t XY = (int64_t)Xs * Ys;
    V->n = 0;
    static const int DYX[9][2] = {{0,0},{0,1},{0,-1},{1,-1},{1,0},{1,1},{-1,-1},{-1,0},{-1,1}};
    int nth = g_nth > 0 ? g_nth : 1;
    if (nU0 < 20000) nth = 1;
    vec64 *loc = (vec64*)calloc(nth, sizeof(vec64));
    size_t blk = (nU0 + nth - 1) / nth;
    #pragma omp parallel num_threads(nth)
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif
        size_t a = (size_t)t * blk, b = a + blk; if (b > nU0) b = nU0;
        for (size_t k = a; k < b; k++) {
            const int64_t u = U0[k];
            const int z = (int)(u / XY); const int64_t rem = u - (int64_t)z * XY;
            const int y = (int)(rem / Xs), x = (int)(rem - (int64_t)y * Xs);
            for (int c = 0; c < 11; c++) {
                int zz = z, yy = y, xx = x;
                if (c < 9) { yy += DYX[c][0]; xx += DYX[c][1]; } else zz += (c == 9 ? -1 : 1);
                if (zz < 0 || zz >= T || yy < 1 || yy > Ys - 2 || xx < 1 || xx > Xs - 2) continue;
                const int64_t w = (int64_t)zz * XY + (int64_t)yy * Xs + xx;
                float bv = bt[w]; if (!(bv > 0.0f && bv <= thr)) continue;
                if (__atomic_exchange_n(&vm[w], (uint8_t)1, __ATOMIC_RELAXED)) continue;     /* already collected */
                v64_push(&loc[t], w);
            }
        }
    }
    size_t tot = 0; for (int t = 0; t < nth; t++) tot += loc[t].n;
    if (tot > V->cap) { V->cap = tot + 1024; V->v = (int64_t*)realloc(V->v, V->cap * sizeof(int64_t)); }
    for (int t = 0; t < nth; t++) { memcpy(V->v + V->n, loc[t].v, loc[t].n * sizeof(int64_t)); V->n += loc[t].n; free(loc[t].v); }
    free(loc);
    for (size_t k = 0; k < V->n; k++) vm[V->v[k]] = 0;          /* (marks only served de-duplication) */
    V->n = sort_list(V->v, V->n, (int64_t)T * XY, g_bm, g_nth);
}

/* ------------------------------------------------------------------ spread (FIFO BFS) */
/* neighbour order of spread_labels_fast: (dz,dy,dx) */
static const int NB[10][3] = {
    { 1, 0, 0}, {-1, 0, 0}, { 0, 0,-1}, { 0, 0, 1}, { 0, 1,-1},
    { 0, 1, 0}, { 0, 1, 1}, { 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1} };

typedef enum { MASK_LT = 0, MASK_LE = 1 } mask_kind;

static inline int mask_ok(float b, float thr, mask_kind mk) { return mk == MASK_LT ? (b < thr) : (b <= thr); }

/* EXACT: tb = int16/100.0 (double);  condition  tb_neigh + delta >= tb_center  (double) */
static inline int grow_ok(int16_t qn, int16_t qc, int delta) {
    return ((double)qn / 100.0 + (double)delta) >= ((double)qc / 100.0);
}

static void spread(const tc_params *P, const float *bt, const int16_t *q, int32_t *lab,
                   float thr, mask_kind mk, int delta, vec64 *queue, int nth, tc_stats *st, double *tacc)
{
    double t0 = now();
    const int T = P->T, Y = P->Y, X = P->X;
    const int64_t XY = (int64_t)X * Y;
    queue->n = 0;
    /* ---- initial queue: labelled voxels with >=1 currently eligible neighbour, raster order ---- */
#ifdef _OPENMP
    if (nth > 1) {
        vec64 *loc = (vec64*)calloc(nth, sizeof(vec64));
        #pragma omp parallel num_threads(nth)
        {
            int t = omp_get_thread_num();
            int64_t z0 = (int64_t)T * t / nth, z1 = (int64_t)T * (t + 1) / nth;
            vec64 *L = &loc[t];
            for (int64_t z = z0; z < z1; z++)
                for (int64_t y = 0; y < Y; y++) {
                    const int64_t row = z * XY + y * X;
                    for (int64_t x = 0; x < X; x++) {
                        const int64_t i = row + x;
                        if (lab[i] <= 0) continue;
                        /* eligibility test */
                        int elig = 0;
                        for (int k = 0; k < 10 && !elig; k++) {
                            int zz = (int)z + NB[k][0], yy = (int)y + NB[k][1], xx = (int)x + NB[k][2];
                            if (zz < 0 || zz >= T || yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
                            int64_t n = (int64_t)zz * XY + (int64_t)yy * X + xx;
                            if (lab[n] != 0 || !mask_ok(bt[n], thr, mk)) continue;
                            if (grow_ok(q[n], q[i], delta)) elig = 1;
                        }
                        if (elig) v64_push(L, i);
                    }
                }
        }
        size_t tot = 0; for (int t = 0; t < nth; t++) tot += loc[t].n;
        if (tot > queue->cap) { queue->cap = tot + 1024; queue->v = (int64_t*)realloc(queue->v, queue->cap * sizeof(int64_t)); }
        for (int t = 0; t < nth; t++) { memcpy(queue->v + queue->n, loc[t].v, loc[t].n * sizeof(int64_t)); queue->n += loc[t].n; free(loc[t].v); }
        free(loc);
    } else
#endif
    {
        for (int64_t z = 0; z < T; z++)
            for (int64_t y = 0; y < Y; y++) {
                const int64_t row = z * XY + y * X;
                for (int64_t x = 0; x < X; x++) {
                    const int64_t i = row + x;
                    if (lab[i] <= 0) continue;
                    int elig = 0;
                    for (int k = 0; k < 10 && !elig; k++) {
                        int zz = (int)z + NB[k][0], yy = (int)y + NB[k][1], xx = (int)x + NB[k][2];
                        if (zz < 0 || zz >= T || yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
                        int64_t n = (int64_t)zz * XY + (int64_t)yy * X + xx;
                        if (lab[n] != 0 || !mask_ok(bt[n], thr, mk)) continue;
                        if (grow_ok(q[n], q[i], delta)) elig = 1;
                    }
                    if (elig) v64_push(queue, i);
                }
            }
    }
    st->n_spread_queue_total += (long long)queue->n;
    double tq = now(); st->t_spread_q += tq - t0;
    /* ---- FIFO BFS ---- */
    size_t head = 0;
    long long claims = 0;
    while (head < queue->n) {
        const int64_t i = queue->v[head++];
        const int z = (int)(i / XY);
        const int64_t rem = i - (int64_t)z * XY;
        const int y = (int)(rem / X);
        const int x = (int)(rem - (int64_t)y * X);
        const int32_t l = lab[i];
        const int16_t qc = q[i];
        for (int k = 0; k < 10; k++) {
            int zz = z + NB[k][0], yy = y + NB[k][1], xx = x + NB[k][2];
            if (zz < 0 || zz >= T || yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
            int64_t n = (int64_t)zz * XY + (int64_t)yy * X + xx;
            if (lab[n] != 0 || !mask_ok(bt[n], thr, mk)) continue;
            if (grow_ok(q[n], qc, delta)) { lab[n] = l; v64_push(queue, n); claims++; }
        }
    }
    st->n_spread_claims_total += claims;
    st->t_spread_bfs += now() - tq;
    *tacc += now() - t0;
}



/* ------------------------------------------------------------------ deterministic parallel BFS */
/* LSD radix sort of uint64 values by the bit field [shift, shift+bits) */
static void sort_u64_field(uint64_t *a, size_t n, int shift, int bits) {
    if (n < 2) return;
    uint64_t *tmp = (uint64_t*)malloc(n * sizeof(uint64_t));
    uint64_t *src = a, *dst = tmp; const int D = 11, R = 1 << D;
    size_t *cnt = (size_t*)malloc(R * sizeof(size_t));
    for (int sh = shift; sh < shift + bits; sh += D) {
        memset(cnt, 0, R * sizeof(size_t));
        for (size_t i = 0; i < n; i++) cnt[(src[i] >> sh) & (R - 1)]++;
        size_t sum = 0; for (int r = 0; r < R; r++) { size_t c = cnt[r]; cnt[r] = sum; sum += c; }
        for (size_t i = 0; i < n; i++) dst[cnt[(src[i] >> sh) & (R - 1)]++] = src[i];
        uint64_t *t = src; src = dst; dst = t;
    }
    if (src != a) memcpy(a, src, n * sizeof(uint64_t));
    free(tmp); free(cnt);
}

#define PAR_MIN_LEVEL 3000
/*
 * Reproduces the FIFO breadth-first labelling of spread() exactly, level by level.
 *   FIFO order == (BFS level, then position of the claiming item in the level, then neighbour index k).
 *   A voxel eligible from several items of one level is claimed by the one with the smallest
 *   key = position*10 + k  (the serial BFS meets that item first).  Edge eligibility (mask, temperature rule) is
 *   static, so claims of different voxels never interact.  Provisional claims are stored as -key in lab[] and
 *   resolved with an atomic "min" (CAS); the next level is the set of winners sorted by key.
 * `queue` holds level 0 on entry; `claimed` returns every newly labelled voxel (unsorted).
 */
static long long bfs_parallel(const tc_params *P, const float *bt, const int16_t *q, int32_t *lab,
                              float thr, mask_kind mk, int delta, vec64 *queue, vec64 *claimed, int nth)
{
    const int T = P->T, Y = P->Y, X = P->X; const int64_t XY = (int64_t)X * Y;
    vec64 cur = {0}, nxt = {0};
    vec64 *cand = (vec64*)calloc(nth, sizeof(vec64));
    vec64 *outp = (vec64*)calloc(nth, sizeof(vec64));          /* (key<<32 | n) packed in int64 */
    cur.v = (int64_t*)malloc((queue->n + 16) * sizeof(int64_t)); cur.cap = queue->n + 16; cur.n = queue->n;
    memcpy(cur.v, queue->v, queue->n * sizeof(int64_t));
    claimed->n = 0;
    long long total = 0;
    while (cur.n > 0) {
        nxt.n = 0;
        if (cur.n < PAR_MIN_LEVEL || (int64_t)cur.n * 10 >= INT32_MAX) {
            for (size_t p = 0; p < cur.n; p++) {
                const int64_t i = cur.v[p];
                const int z = (int)(i / XY); const int64_t rem = i - (int64_t)z * XY;
                const int y = (int)(rem / X), x = (int)(rem - (int64_t)y * X);
                const int32_t l = lab[i]; const int16_t qc = q[i];
                for (int k = 0; k < 10; k++) {
                    int zz = z + NB[k][0], yy = y + NB[k][1], xx = x + NB[k][2];
                    if (zz < 0 || zz >= T || yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
                    int64_t n = (int64_t)zz * XY + (int64_t)yy * X + xx;
                    if (lab[n] != 0 || !mask_ok(bt[n], thr, mk)) continue;
                    if (grow_ok(q[n], qc, delta)) { lab[n] = l; v64_push(&nxt, n); }
                }
            }
        } else {
            for (int t = 0; t < nth; t++) { cand[t].n = 0; outp[t].n = 0; }
            const size_t m = cur.n, blk = (m + nth - 1) / nth;
            #pragma omp parallel num_threads(nth)
            {
#ifdef _OPENMP
                int t = omp_get_thread_num();
#else
                int t = 0;
#endif
                size_t a = (size_t)t * blk, b = a + blk; if (b > m) b = m;
                for (size_t p = a; p < b; p++) {
                    const int64_t i = cur.v[p];
                    const int z = (int)(i / XY); const int64_t rem = i - (int64_t)z * XY;
                    const int y = (int)(rem / X), x = (int)(rem - (int64_t)y * X);
                    const int16_t qc = q[i];
                    for (int k = 0; k < 10; k++) {
                        int zz = z + NB[k][0], yy = y + NB[k][1], xx = x + NB[k][2];
                        if (zz < 0 || zz >= T || yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
                        int64_t n = (int64_t)zz * XY + (int64_t)yy * X + xx;
                        int32_t old = __atomic_load_n(&lab[n], __ATOMIC_RELAXED);
                        if (old > 0) continue;
                        if (!mask_ok(bt[n], thr, mk) || !grow_ok(q[n], qc, delta)) continue;
                        const int32_t key = (int32_t)(p * 10 + k + 1);
                        for (;;) {
                            if (old > 0) break;
                            if (old < 0 && -old <= key) break;
                            if (__atomic_compare_exchange_n(&lab[n], &old, -key, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { v64_push(&cand[t], n); break; }
                        }
                    }
                }
            }
            #pragma omp parallel num_threads(nth)
            {
#ifdef _OPENMP
                int t = omp_get_thread_num();
#else
                int t = 0;
#endif
                for (size_t j = 0; j < cand[t].n; j++) {
                    const int64_t n = cand[t].v[j];
                    int32_t v = __atomic_load_n(&lab[n], __ATOMIC_RELAXED);
                    if (v >= 0) continue;
                    const int32_t key = -v;
                    const int64_t c = cur.v[(key - 1) / 10];
                    const int32_t nl = lab[c];
                    if (__atomic_compare_exchange_n(&lab[n], &v, nl, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                        v64_push(&outp[t], (int64_t)(((uint64_t)(uint32_t)key << 32) | (uint64_t)n));
                }
            }
            size_t tot = 0; for (int t = 0; t < nth; t++) tot += outp[t].n;
            uint64_t *buf = (uint64_t*)malloc((tot + 1) * sizeof(uint64_t)); size_t w = 0;
            for (int t = 0; t < nth; t++) { memcpy(buf + w, outp[t].v, outp[t].n * sizeof(uint64_t)); w += outp[t].n; }
            sort_u64_field(buf, tot, 32, 33);
            if (tot + 16 > nxt.cap) { nxt.cap = tot + 16; nxt.v = (int64_t*)realloc(nxt.v, nxt.cap * sizeof(int64_t)); }
            for (size_t j = 0; j < tot; j++) nxt.v[j] = (int64_t)(buf[j] & 0xFFFFFFFFu);
            nxt.n = tot; free(buf);
        }
        /* record claims, advance a level */
        if (claimed->n + nxt.n > claimed->cap) { claimed->cap = claimed->n + nxt.n + 1024; claimed->v = (int64_t*)realloc(claimed->v, claimed->cap * sizeof(int64_t)); }
        memcpy(claimed->v + claimed->n, nxt.v, nxt.n * sizeof(int64_t)); claimed->n += nxt.n;
        total += (long long)nxt.n;
        vec64 tmp = cur; cur = nxt; nxt = tmp;
    }
    for (int t = 0; t < nth; t++) { free(cand[t].v); free(outp[t].v); }
    free(cand); free(outp); free(cur.v); free(nxt.v);
    return total;
}

/* spread() with a maintained frontier-candidate list F (sorted, unique, superset of the labelled voxels that
   have at least one unlabelled in-bounds neighbour).  Same FIFO order as the literal version. */
static void spread_fast(const tc_params *P, const float *bt, const int16_t *q, int32_t *lab,
                        float thr, mask_kind mk, int delta, vec64 *queue, vec64 *F, vec64 *scratch,
                        int nth, tc_stats *st, double *tacc)
{
    double t0 = now();
    const int T = P->T, Y = P->Y, X = P->X;
    const int64_t XY = (int64_t)X * Y;
    const size_t nF = F->n;
    if (nth < 1) nth = 1;
    vec64 *kl = (vec64*)calloc(nth, sizeof(vec64)), *ql = (vec64*)calloc(nth, sizeof(vec64));
    size_t blk = (nF + nth - 1) / nth;
    #pragma omp parallel num_threads(nth)
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif
        size_t a = (size_t)t * blk, b = a + blk; if (b > nF) b = nF;
        for (size_t k = a; k < b; k++) {
            const int64_t i = F->v[k];
            if (lab[i] <= 0) continue;
            const int z = (int)(i / XY); const int64_t rem = i - (int64_t)z * XY;
            const int y = (int)(rem / X), x = (int)(rem - (int64_t)y * X);
            int u = 0, e = 0;
            for (int d = 0; d < 10; d++) {
                int zz = z + NB[d][0], yy = y + NB[d][1], xx = x + NB[d][2];
                if (zz < 0 || zz >= T || yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
                int64_t n = (int64_t)zz * XY + (int64_t)yy * X + xx;
                if (lab[n] != 0) continue;
                u = 1;
                if (mask_ok(bt[n], thr, mk) && grow_ok(q[n], q[i], delta)) { e = 1; break; }
            }
            if (u) v64_push(&kl[t], i);
            if (e) v64_push(&ql[t], i);
        }
    }
    size_t totk = 0, totq = 0;
    for (int t = 0; t < nth; t++) { totk += kl[t].n; totq += ql[t].n; }
    if (totk > F->cap) { F->cap = totk + 1024; F->v = (int64_t*)realloc(F->v, F->cap * sizeof(int64_t)); }
    if (totq + 1024 > queue->cap) { queue->cap = totq + 1024; queue->v = (int64_t*)realloc(queue->v, queue->cap * sizeof(int64_t)); }
    F->n = 0; queue->n = 0;
    for (int t = 0; t < nth; t++) {
        memcpy(F->v + F->n, kl[t].v, kl[t].n * sizeof(int64_t)); F->n += kl[t].n;
        memcpy(queue->v + queue->n, ql[t].v, ql[t].n * sizeof(int64_t)); queue->n += ql[t].n;
        free(kl[t].v); free(ql[t].v);
    }
    free(kl); free(ql);
    st->n_spread_queue_total += (long long)queue->n;
    double tq = now(); st->t_spread_q += tq - t0;
    long long claims = 0;
    if (P->fast >= 2 && nth > 1 && (int64_t)T * XY < ((int64_t)1 << 32)) {
        vec64 cl = {0};
        claims = bfs_parallel(P, bt, q, lab, thr, mk, delta, queue, &cl, nth);
        st->n_spread_claims_total += claims;
        if (cl.n) {
            cl.n = sort_list(cl.v, cl.n, (int64_t)T * XY, g_bm, nth);
            merge_unique(F->v, F->n, cl.v, cl.n, F, scratch);
        }
        free(cl.v);
    } else {
        const size_t n0 = queue->n;
        size_t head = 0;
        while (head < queue->n) {
            const int64_t i = queue->v[head++];
            const int z = (int)(i / XY); const int64_t rem = i - (int64_t)z * XY;
            const int y = (int)(rem / X), x = (int)(rem - (int64_t)y * X);
            const int32_t l = lab[i]; const int16_t qc = q[i];
            for (int k = 0; k < 10; k++) {
                int zz = z + NB[k][0], yy = y + NB[k][1], xx = x + NB[k][2];
                if (zz < 0 || zz >= T || yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
                int64_t n = (int64_t)zz * XY + (int64_t)yy * X + xx;
                if (lab[n] != 0 || !mask_ok(bt[n], thr, mk)) continue;
                if (grow_ok(q[n], qc, delta)) { lab[n] = l; v64_push(queue, n); claims++; }
            }
        }
        st->n_spread_claims_total += claims;
        if (queue->n > n0) {
            size_t nc = queue->n - n0;
            nc = sort_list(queue->v + n0, nc, (int64_t)T * XY, g_bm, nth);
            merge_unique(F->v, F->n, queue->v + n0, nc, F, scratch);
        }
    }
    st->t_spread_bfs += now() - tq;
    *tacc += now() - t0;
}

/* after label_region: assigned voxels (positive: new frontier candidates; zeroed: their labelled neighbours) */
static void front_after_label(const tc_params *P, const int32_t *lab, const vec64 *asg, vec64 *F, vec64 *scratch, vec64 *tmp) {
    const int T = P->T, Y = P->Y, X = P->X; const int64_t XY = (int64_t)X * Y;
    tmp->n = 0;
    for (size_t k = 0; k < asg->n; k++) {
        const int64_t i = asg->v[k];
        if (lab[i] > 0) { v64_push(tmp, i); continue; }
        const int z = (int)(i / XY); const int64_t rem = i - (int64_t)z * XY;
        const int y = (int)(rem / X), x = (int)(rem - (int64_t)y * X);
        for (int d = 0; d < 10; d++) {
            int zz = z + NB[d][0], yy = y + NB[d][1], xx = x + NB[d][2];
            if (zz < 0 || zz >= T || yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
            int64_t n = (int64_t)zz * XY + (int64_t)yy * X + xx;
            if (lab[n] > 0) v64_push(tmp, n);
        }
    }
    if (tmp->n) {
        tmp->n = sort_list(tmp->v, tmp->n, (int64_t)T * XY, g_bm, g_nth);
        merge_unique(F->v, F->n, tmp->v, tmp->n, F, scratch);
    }
}

/* initial candidate list: all labelled voxels with an unlabelled neighbour (ascending) */
static void front_init(const tc_params *P, const int32_t *lab, vec64 *F, int nth) {
    const int T = P->T, Y = P->Y, X = P->X; const int64_t XY = (int64_t)X * Y;
    F->n = 0;
    vec64 *loc = (vec64*)calloc(nth, sizeof(vec64));
    #pragma omp parallel num_threads(nth)
    {
#ifdef _OPENMP
        int t = omp_get_thread_num();
#else
        int t = 0;
#endif
        int64_t z0 = (int64_t)T * t / nth, z1 = (int64_t)T * (t + 1) / nth;
        for (int64_t z = z0; z < z1; z++) for (int64_t y = 0; y < Y; y++) for (int64_t x = 0; x < X; x++) {
            const int64_t i = z * XY + y * X + x;
            if (lab[i] <= 0) continue;
            int u = 0;
            for (int d = 0; d < 10 && !u; d++) {
                int zz = (int)z + NB[d][0], yy = (int)y + NB[d][1], xx = (int)x + NB[d][2];
                if (zz < 0 || zz >= T || yy < 0 || yy >= Y || xx < 0 || xx >= X) continue;
                if (lab[(int64_t)zz * XY + (int64_t)yy * X + xx] == 0) u = 1;
            }
            if (u) v64_push(&loc[t], i);
        }
    }
    for (int t = 0; t < nth; t++) {
        if (F->n + loc[t].n > F->cap) { F->cap = F->n + loc[t].n + 1024; F->v = (int64_t*)realloc(F->v, F->cap * sizeof(int64_t)); }
        memcpy(F->v + F->n, loc[t].v, loc[t].n * sizeof(int64_t)); F->n += loc[t].n; free(loc[t].v);
    }
    free(loc);
}

/* ------------------------------------------------------------------ label_region */
/* ope[] construction; returns number of labelled operator voxels (Nb_pixNeighbors). */
static inline int make_operator(const tc_params *P, const int32_t *lab, int64_t ope[11], int X, int Y, int Z) {
    const int64_t XY = (int64_t)P->X * P->Y, XS = P->X;
    const int64_t i0 = Z * XY + (int64_t)Y * XS + X;
    int n = 0;
    ope[0] = i0;                      if (lab[i0] > 0) n++;
    ope[1] = i0 + 1;                  if (lab[ope[1]] > 0) n++;           /* x+1        */
    ope[2] = i0 - 1;                  if (lab[ope[2]] > 0) n++;           /* x-1        */
    ope[3] = i0 + XS - 1;             if (lab[ope[3]] > 0) n++;           /* x-1,y+1    */
    ope[4] = i0 + XS;                 if (lab[ope[4]] > 0) n++;           /* y+1        */
    ope[5] = i0 + XS + 1;             if (lab[ope[5]] > 0) n++;           /* x+1,y+1    */
    ope[6] = i0 - XS - 1;             if (lab[ope[6]] > 0) n++;           /* x-1,y-1    */
    ope[7] = i0 - XS;                 if (lab[ope[7]] > 0) n++;           /* y-1        */
    ope[8] = i0 - XS + 1;             if (lab[ope[8]] > 0) n++;           /* x+1,y-1    */
    if (Z == 0) { ope[9] = i0; }                                          /* F_operator10_Z1 */
    else        { ope[9] = i0 - XY;  if (lab[ope[9]] > 0) n++; }
    if (Z == P->T - 1) { ope[10] = i0; }                                  /* F_operator10_ZFINAL */
    else               { ope[10] = i0 + XY; if (lab[ope[10]] > 0) n++; }
    return n;
}

/* Ordered distinct positive labels (uniq() in label.c); returns count. */
static inline int uniq11(const int32_t *lab, const int64_t ope[11], int out[11]) {
    int m = 0;
    for (int k = 0; k < 11; k++) {
        int l = lab[ope[k]];
        if (l > 0) {
            int dup = 0;
            for (int j = 0; j < m; j++) if (out[j] == l) { dup = 1; break; }
            if (!dup) out[m++] = l;
        }
    }
    return m;
}

static int voisin_add(Ent *e, int target) {
    for (int k = 0; k < e->nb_neighbours; k++) if (e->voisin[k] == target) return 0;
    if (e->nb_neighbours >= NBMAX_NEIGHBORED_OBJECTS) return 0;       /* list full: original silently drops */
    if (e->nb_neighbours >= e->vcap) {
        int nc = e->vcap ? e->vcap * 2 : 8; if (nc > NBMAX_NEIGHBORED_OBJECTS) nc = NBMAX_NEIGHBORED_OBJECTS;
        e->voisin = (int*)realloc(e->voisin, nc * sizeof(int)); e->vcap = nc;
    }
    e->voisin[e->nb_neighbours++] = target;
    return 0;
}

/* Iterative, order-exact version of Relabel() (original is recursive DFS). */
typedef struct { long ind; int i; } relframe;
static int relabel(EntTab *t, long ind0, int label, int labelMin) {
    Ent *E;
    long root = (long)label - 1 - labelMin;
    size_t cap = 256, sp = 0;
    relframe *st = (relframe*)malloc(cap * sizeof(relframe));
    if (t->e[ind0].flagRelabel != 0) { free(st); return 0; }
    t->e[ind0].flagRelabel = 2;
    st[sp].ind = ind0; st[sp].i = 0; sp++;
    while (sp) {
        relframe *f = &st[sp - 1];
        E = t->e;
        if (f->i < E[f->ind].nb_neighbours) {
            int tmp = E[f->ind].voisin[f->i++];
            long ti = (long)tmp - 1 - labelMin;
            if (ent_ensure(t, ti)) { free(st); return -1; }
            E = t->e;
            if (E[ti].flagFIX == 1) {
                E[root].flagRelabel = 1;
                E[root].labelMCS_alreadyidentified = E[ti].label;
            }
            if (E[ti].flagRelabel == 0) {
                if (E[ti].label != label && E[ti].flagFIX != 1) {
                    E[ti].label = label;
                    E[ti].flagRelabel = 2;                    /* entry of the recursive call */
                    if (sp == cap) { cap *= 2; st = (relframe*)realloc(st, cap * sizeof(relframe)); }
                    st[sp].ind = ti; st[sp].i = 0; sp++;
                }
            }
        } else {
            long done = f->ind; sp--;
            if (sp) t->e[done].flagRelabel = -1;              /* caller: p[tmp].flagRelabel = -1 */
        }
    }
    free(st);
    return 0;
}

/* One call of Label_region().  Returns updated NSEEDS, or <0 on error. */
static int label_region(const tc_params *P, tc_state *S, const float *bt, int32_t *lab, const float *area2d,
                        const vec64 *scan, vec64 *post, vec64 *asg, float thr, int NSEEDS, int labelMin, tc_stats *st)
{
    const int fast = P->fast;
    if (fast) asg->n = 0;
    const int T = P->T, Ys = P->Y, Xs = P->X;
    const int64_t XY = (int64_t)Xs * Ys;
    EntTab tab; memset(&tab, 0, sizeof(tab)); tab.T = T;
    double t0 = now();

    /* ptr_seed[i] <- p[i] for i < nbMaxCluster with p[i].label>0 (flagFIX = 1) */
    long hi = -1;
    for (long i = 0; i < S->cap && i < P->nbMaxCluster; i++) {
        if (S->label[i] > 0) {
            if (ent_ensure(&tab, i)) { ent_free_all(&tab); return -2; }
            Ent *e = &tab.e[i];
            e->label = S->label[i]; e->flagFIX = 1; e->flagRelabel = 2;
            e->seed_area = 0x3fffffff;          /* always >= timin*lifemin (invariant of fixed objects) */
            e->seed_duration = 0x3fffffff;
            if (i > hi) hi = i;
        }
    }

    int number = NSEEDS;
    int64_t ope[11]; int labs[11];
    long long nraw = 0;

    const float thrf = thr;
    /* ---------------- scan of the cold voxels, ascending index order ---------------- */
    for (size_t k = 0; k < scan->n; k++) {
        const int64_t i = scan->v[k];
        const int Z = (int)(i / XY);
        const int64_t rem = i - (int64_t)Z * XY;
        const int Yc = (int)(rem / Xs);
        const int Xc = (int)(rem - (int64_t)Yc * Xs);
        if (Xc == 0 || Xc == Xs - 1 || Yc == 0 || Yc == Ys - 1) continue;
        int Nb = make_operator(P, lab, ope, Xc, Yc, Z);
        /* isolated segmented pixel: operator entries 1..6 all not cold */
        {
            int iso = 1;
            for (int j = 1; j <= 6; j++) { float b = bt[ope[j]]; if (b > 0.0f && b <= thrf) { iso = 0; break; } }
            if (iso) continue;
        }
        int label_object;
        if (Nb == 0) {
            number++;
            label_object = number;
            lab[i] = number;
            if (fast) v64_push(asg, i);
            long idx = (long)label_object - 1 - labelMin;
            if (idx >= NBMAX_LABEL_OBJECTS) { ent_free_all(&tab); return -3; }   /* original: exit(0) */
            if (ent_ensure(&tab, idx)) { ent_free_all(&tab); return -3; }
            tab.e[idx].label = label_object;
            if (idx > hi) hi = idx;
            nraw++;
        }
        int m = uniq11(lab, ope, labs);
        label_object = labs[0];
        long lidx = (long)label_object - 1 - labelMin;
        if (ent_ensure(&tab, lidx)) { ent_free_all(&tab); return -4; }
        Ent *Eo = &tab.e[lidx];
        if (m <= 1 && Eo->flagFIX <= 0) {
            for (int c = 0; c < 11; c++) {
                float b = bt[ope[c]];
                if ((b > 0.0f && b <= thrf) && lab[ope[c]] == 0) {
                    lab[ope[c]] = label_object;
                    if (fast) v64_push(asg, ope[c]);
                    Eo->label = label_object;
                }
            }
        }
        if (m > 1) {
            for (int c = 0; c < 11; c++) {
                float b = bt[ope[c]];
                if ((b > 0.0f && b <= thrf) && lab[ope[c]] == 0) {
                    if (Eo->flagFIX <= 0) { lab[ope[c]] = label_object; if (fast) v64_push(asg, ope[c]); }
                }
            }
            for (int c = 0; c < m; c++) {
                long aidx = (long)labs[c] - 1 - labelMin;
                if (ent_ensure(&tab, aidx)) { ent_free_all(&tab); return -4; }
                voisin_add(&tab.e[lidx], labs[c]);
                voisin_add(&tab.e[aidx], label_object);
            }
        }
    }
    st->t_scan += now() - t0; t0 = now();
    st->n_raw_labels_total += nraw;

    /* ---------------- regroup touching, non-fixed objects ---------------- */
    for (long i = 0; i < tab.n; i++) {
        Ent *e = &tab.e[i];
        if (e->flagRelabel == 0 && e->nb_neighbours > 0 && e->label > 0 && e->flagFIX != 1) {
            if (relabel(&tab, i, e->label, labelMin)) { ent_free_all(&tab); return -5; }
            e = &tab.e[i];
        }
        e = &tab.e[i];
        if (e->flagRelabel >= 0 && e->nb_neighbours == 0 && e->label > 0 && e->flagFIX != 1) e->flagRelabel = 2;
    }
    st->t_merge += now() - t0; t0 = now();

    /* ---------------- per-object area / per-frame area (EXACT int = int + float) ---------------- */
    if (fast) {
        /* only voxels that received a (transient) label in this call can change; ascending order is
           required so that the float accumulation order equals the original one */
        asg->n = sort_list(asg->v, asg->n, (int64_t)T * XY, g_bm, g_nth);
        post = asg;
    }
    st->n_asg_total += (long long)(fast ? asg->n : 0);
    for (size_t k = 0; k < post->n; k++) {
        const int64_t i = post->v[k];
        if (lab[i] > 0) {
            const int Z = (int)(i / XY);
            const int64_t rem = i - (int64_t)Z * XY;
            const int64_t idx2d = rem;                       /* = X + Y*XSIZE */
            long eidx = (long)lab[i] - 1 - labelMin;
            if (eidx < 0 || ent_ensure(&tab, eidx)) { ent_free_all(&tab); return -6; }
            lab[i] = tab.e[eidx].label;
            long nidx = (long)lab[i] - 1 - labelMin;
            if (nidx < 0 || ent_ensure(&tab, nidx)) { ent_free_all(&tab); return -6; }
            Ent *e = &tab.e[nidx];
            if (e->flagFIX == 1) continue;                   /* fixed objects: values never used by any decision */
            if (!e->perframe) e->perframe = (int*)calloc(T, sizeof(int));
            float a = area2d[idx2d];
            e->perframe[Z] = e->perframe[Z] + a;             /* int = int + float : float add, then truncation */
            e->seed_area   = e->seed_area + a;
            if (Z < e->slotBegin) e->slotBegin = Z;
            if (Z > e->slotEnd)   e->slotEnd = Z;
            if (nidx > hi) hi = nidx;
        }
    }
    st->t_accum += now() - t0; t0 = now();

    /* ---------------- lifetime / size criteria ---------------- */
    for (long i = 0; i < tab.n; i++) {
        Ent *e = &tab.e[i];
        if (e->flagFIX != 1) {
            if (e->flagRelabel < 1) { e->label = 0; e->flagFIX = 0; continue; }
            if ((e->slotEnd - e->slotBegin) + 1 < P->lifemin) { e->label = 0; e->flagFIX = 0; continue; }
            for (int j = e->slotBegin; j <= e->slotEnd; j++) {
                int pf = e->perframe ? e->perframe[j] : 0;
                if (e->seed_duration < P->lifemin && pf < P->timin && e->flagFIX != 1) e->seed_duration = 0;
                if (e->label != 0 && pf >= P->timin && e->flagFIX != 1) e->seed_duration = e->seed_duration + 1;
            }
        }
    }
    st->t_filter += now() - t0; t0 = now();

    /* ---------------- renumber the surviving objects ---------------- */
    int n = NSEEDS + 1;
    int newNS = NSEEDS;
    long long nnew = 0;
    const int areamin = P->timin * P->lifemin;
    for (long i = 0; i < tab.n; i++) {
        Ent *e = &tab.e[i];
        if (e->flagFIX != 1) {
            if (e->label != 0 && e->seed_area >= areamin && e->seed_duration >= P->lifemin) {
                e->label = n;
                long pidx = (long)n - labelMin - 1;
                if (pidx >= P->nbMaxCluster) { ent_free_all(&tab); return -7; }   /* original: heap overflow of p[] */
                if (pfix_set(S, pidx, n)) { ent_free_all(&tab); return -7; }
                if (n > newNS) newNS = n;
                nnew++;
                n++;
            }
        }
    }
    st->n_new_mcs_total += nnew;

    /* ---------------- renumber the image labels ---------------- */
    for (size_t k = 0; k < post->n; k++) {
        const int64_t i = post->v[k];
        if (lab[i] > labelMin) {
            long eidx = (long)lab[i] - 1 - labelMin;
            if (eidx >= tab.n) { lab[i] = 0; continue; }     /* original reads a zeroed calloc'd Blob -> label 0 */
            Ent *e = &tab.e[eidx];
            int fixed_ok = (e->flagFIX == 1);
            if (e->label != 0 && (fixed_ok || (e->seed_area >= areamin && e->seed_duration >= P->lifemin))) {
                lab[i] = e->label;
            } else {
                if (e->flagRelabel == 1) lab[i] = e->labelMCS_alreadyidentified;
                else lab[i] = 0;
            }
        }
    }
    st->t_remap += now() - t0;
    ent_free_all(&tab);
    return newNS;
}


/* mask_new ⊆ mask_closed ?   (masks are {bt < L} or {bt <= L}) */
static int mask_subset(mask_kind nk, float nL, mask_kind ck, float cL) {
    if (ck == MASK_LE) return nL <= cL;                 /* {<a} or {<=a}  ⊆ {<=b}  iff  a <= b */
    return nk == MASK_LT ? (nL <= cL) : (nL < cL);       /* closed is {<b}: {<a} iff a<=b ; {<=a} iff a<b */
}

/* ------------------------------------------------------------------ public entry point */
/*
 * Runs the whole threshold loop of detect_and_spread() on one chunk, in place.
 *   bt     : float32 (T,Y,X)  brightness temperature (K)
 *   lab    : int32   (T,Y,X)  in: labels of the overlap slots (others 0); out: final labels
 *   area2d : float32 (Y,X)    pixel area (km2)
 * returns the new nb_ConvSeeds (>=0) or a negative error code.
 */
int tc_run_chunk(tc_state *S, const tc_params *P, const float *bt, const float *area2d, int32_t *lab,
                 int nseeds_in, int labelMin, int nthreads, tc_stats *st)
{
    memset(st, 0, sizeof(*st));
    const int64_t nvox = (int64_t)P->T * P->Y * P->X;
    int nth = nthreads > 0 ? nthreads : 1;
    int16_t *q = (int16_t*)malloc(nvox * sizeof(int16_t));
    if (!q) return -100;
    double t0 = now();
    /* EXACT quantisation used by the numba kernel: (volume_bt*100).astype(int16) */
    #pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t i = 0; i < nvox; i++) {
        float b = bt[i];
        if (b != b) { q[i] = INT16_MIN; continue; }
        if (P->quant_mode == 0) { double d = (double)b * 100.0; q[i] = (int16_t)(int32_t)d; }
        else { float f = b * 100.0f; q[i] = (int16_t)(int32_t)f; }
    }
    st->t_quant = now() - t0;

    g_nth = nth; g_bm = (uint64_t*)calloc((size_t)(nvox / 64 + 2), sizeof(uint64_t));
    vec64 queue = {0}, cold = {0}, Ulist = {0}, Unew = {0}, Utmp = {0}, V = {0}, asg = {0}, Fl = {0}, Fscr = {0}, Ftmp = {0};
    tc_bins bins = {0}; uint8_t *vm = NULL;
    if (P->fast) {
        double tb0 = now();
        build_bins(P, bt, nvox, &bins, nth);
        vm = (uint8_t*)calloc((size_t)nvox, 1);
        st->t_bins = now() - tb0;
        front_init(P, lab, &Fl, nth);
    }
    int NS = nseeds_in;
    int thr_last_next = P->maxBT;
    int kthr = 0;
    int cl_valid = 0; mask_kind cl_k = MASK_LE; float cl_L = 0.0f; int cl_d = 0;   /* closure of the last SPREAD */
    for (int thr = P->minBT; thr < P->maxBT + P->stepBT; thr += P->stepBT, kthr++) {
        int next = thr + P->stepBT; if (next > P->maxBT) next = P->maxBT;
        thr_last_next = next;
        /* step 1 */
        if (P->fast) {
            /* After a SPREAD the labelling is CLOSED: no labelled voxel has an eligible unlabelled neighbour for that
               mask/delta.  If nothing changed since and the new mask is a subset, the BFS frontier is empty. */
            if (cl_valid && cl_d == P->delta_spread && mask_subset(MASK_LT, (float)thr, cl_k, cl_L)) st->n_spread_skipped++;
            else spread_fast(P, bt, q, lab, (float)thr, MASK_LT, P->delta_spread, &queue, &Fl, &Fscr, nth, st, &st->t_spread1);
        } else spread(P, bt, q, lab, (float)thr, MASK_LT, P->delta_spread, &queue, nth, st, &st->t_spread1);
        /* step 2 */
        int r;
        if (!P->fast) {
            double t1 = now();
            build_cold_list(bt, nvox, (float)thr, &cold, nth);
            st->t_coldlist += now() - t1;
            st->n_cold_total += (long long)cold.n; st->n_label_calls++;
            r = label_region(P, S, bt, lab, area2d, &cold, &cold, &asg, (float)thr, NS, labelMin, st);
        } else {
            double t1 = now();
            /* U0 = {cold, unlabelled at call start} : previous U minus newly labelled, plus voxels newly cold */
            pack_unlabelled(Ulist.v, Ulist.n, lab, &Utmp, nth);
            pack_unlabelled(bins.idx + bins.off[kthr], (size_t)(bins.off[kthr + 1] - bins.off[kthr]), lab, &Unew, nth);
            merge_sorted(Utmp.v, Utmp.n, Unew.v, Unew.n, &Ulist);
            st->t_ulist += now() - t1; t1 = now();
            build_V(P, bt, (float)thr, Ulist.v, Ulist.n, vm, &V);
            st->t_vset += now() - t1;
            st->n_U0_total += (long long)Ulist.n; st->n_V_total += (long long)V.n;
            st->n_cold_total += (long long)V.n; st->n_label_calls++;
            r = label_region(P, S, bt, lab, area2d, &V, &V, &asg, (float)thr, NS, labelMin, st);
        }
        cl_valid = 0;                                   /* label_region changed the labels */
        if (r < 0) { st->err = r; free(q); free(queue.v); free(cold.v); free(Ulist.v); free(Unew.v); free(Utmp.v); free(V.v); free(asg.v); free(vm); free(bins.off); free(bins.idx); free(Fl.v); free(Fscr.v); free(Ftmp.v); free(g_bm); g_bm = NULL; return r; }
        NS = r;
        /* step 3 */
        if (P->fast) {
            front_after_label(P, lab, &asg, &Fl, &Fscr, &Ftmp);
            spread_fast(P, bt, q, lab, (float)next, MASK_LE, P->delta_spread, &queue, &Fl, &Fscr, nth, st, &st->t_spread2);
            cl_valid = 1; cl_k = MASK_LE; cl_L = (float)next; cl_d = P->delta_spread;
        } else spread(P, bt, q, lab, (float)next, MASK_LE, P->delta_spread, &queue, nth, st, &st->t_spread2);
    }
    if (P->fast) spread_fast(P, bt, q, lab, (float)thr_last_next, MASK_LE, P->final_delta, &queue, &Fl, &Fscr, nth, st, &st->t_spread3);
    else spread(P, bt, q, lab, (float)thr_last_next, MASK_LE, P->final_delta, &queue, nth, st, &st->t_spread3);
    free(q); free(queue.v); free(cold.v); free(Ulist.v); free(Unew.v); free(Utmp.v); free(V.v); free(asg.v); free(vm); free(bins.off); free(bins.idx); free(Fl.v); free(Fscr.v); free(Ftmp.v); free(g_bm); g_bm = NULL;
    return NS;
}

/* multi-threaded zeroing of a large int32 buffer (the label volume of the next chunk) */
void tc_zero_i32(int32_t *a, int64_t n, int nthreads) {
    int nth = nthreads > 0 ? nthreads : 1;
    #pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t i = 0; i < n; i++) a[i] = 0;
}

int tc_stats_size(void) { return (int)sizeof(tc_stats); }
int tc_params_size(void) { return (int)sizeof(tc_params); }
