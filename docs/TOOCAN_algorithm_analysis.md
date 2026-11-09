# TOOCAN 算法分析（基于源码审查）

审查对象：`toocan-python`（https://gitlab.in2p3.fr/thomas.fiolleau1/toocan-python）（TOOCAN v2.08 的 Python 实现，核心为 C + Numba），
以及作者在 DKRZ Levante 上使用的两个 SLURM 运行脚本（未随本仓库发布）。以下所有结论均来自对源码的逐行阅读，
并通过 `TOOCAN_performance_report.md` §3 的逐位对照（同一输入、同一参数，生产规模）交叉验证：凡文中断言的语义，都被新实现复现到了每一个像素的标签值。

> **重要更正（与任务描述不同之处）**：该代码并非"纯 C/Fortran 程序"。它是 **Python 驱动 + 一个 C 函数
> （`label.c: Label_region`，通过 ctypes 调用）+ 一个 Numba 函数（`spreading.py: spread_labels_fast`）**。
> 没有 Fortran，没有 OpenMP，没有 MPI，没有多线程（见第 7 节）。

## 1. 程序入口与调用关系

```
python -m toocan.run_toocan  fileparam_TOOCAN.dat fileparam_GEO.dat  Y M D h m  Y M D h m  lonmin lonmax latmin latmax
   (toocan/run_toocan.py —— 脚本式驱动，模块导入时即解析 sys.argv，无 main())
 ├─ parse_param_file()                       lib/utils/parse_param_file.py
 ├─ crop_navigation_file()                   lib/io/open_navigationFile.py   → 像素面积 mat_surfacePix、经纬度
 ├─ build_ir_filelist(expand_time_records)   lib/io/file_listing.py          → [(文件, 时间, time_index)]
 └─ while current_start < end_global:        ←←← 逐块（chunk）串行循环
      ├─ 处理重叠区标签延续 + clean_clusters()   lib/detection_spreading/cluster_tools.py
      ├─ extract_volume()  → read_ir_file() × 72   lib/io/open_IRdata.py    （每个时次单独 xr.open_dataset）
      ├─ detect_and_spread()                    lib/detection_spreading/detect_and_spread.py   ★核心
      │    └─ for thr in 190,192,…,236:
      │         ├─ spread_labels_fast(mask = BT <  thr)          (Numba, 单线程 BFS)
      │         ├─ detect_objects() → prepare_cloudy_indices() + liblabel.Label_region()   (C)
      │         ├─ list_clusters()   (仅打印，调用 np.unique)
      │         └─ spread_labels_fast(mask = BT <= thr_next)
      │       最后再 spread_labels_fast(mask = BT <= 235, delta = 10)
      └─ save_labels_slot_by_slot() → save_labels_daily()   每日一个 NetCDF，合并重叠时次
后处理（独立入口）：python -m toocan.run_write_filetracking …
   └─ track_month_from_df()  lib/postprocessing/tracking/tracking_DCS.py  → INT/LC 变量 → NetCDF
```

## 2. 输入数据要求

| 项 | 内容 |
|---|---|
| 变量 | 云顶亮温 BT（K）。本项目用 ICON 模式的 `rlut`（OLR，W m⁻²），由 `convert_olr_to_irbt` 转为 BT：`Tb=(OLR/σ)^¼`，`BT=0.00443828·Tb²−0.543846·Tb+129.544` |
| 格式 | NetCDF4，每日一个文件，含 72 个时次（20 分钟分辨率）；`rlut(time,lat,lon)` float32，zlib 压缩，chunk=(1,72,3600) |
| 网格 | 0.1°，601×3600（30°S–30°N，0–359.9°E），`navigation_C5_tropics_0p1deg.nc` 提供像素面积 `mat_surfacePix` |
| 内存布局 | (T, Y, X) C 连续；线性索引 `i = z·Y·X + y·X + x` |
| 缺测 | NaN；`BT>0` 与 `BT<阈值` 均对 NaN 为假，故缺测像素既不参与种子检测也不参与扩散 |
| 时间 | 连续、等间隔；块内缺时次 → 驱动程序 `sys.exit` 报错（仅允许最后一块不满） |

## 3. 参数及其科学含义（`config/fileparam_TOOCAN_C5_*_1980.dat`）

| 参数 | 值 | 含义 |
|---|---|---|
| `minBT_threshold` / `maxBT_threshold` / `stepBT_threshold` | 190 / 235 / 2 K | 层次化阈值序列。**注意：循环是 `range(190, 235+2, 2)`，实际包含 236 K**，其 `threshold_next=min(238,235)=235` |
| `deltaBT_Spread` | 1 K | 区域生长时，允许把比中心像元"冷最多 1 K"的像元并入（BT_邻 + Δ ≥ BT_中）；即不向显著更冷的方向"下坡"生长 |
| 最终 spread 的 Δ | 10 K（硬编码） | 最后一次生长放宽到 10 K |
| `minAreaSeed` | 625 km² | 每个时次内对象面积下限（`timin`） |
| `minLifetime` | 3 个时次 | 最短持续时间（`lifemin`），且要求**连续** ≥3 个时次面积≥625 km² |
| 面积×寿命下限 | 625×3=1875 | `seed_area ≥ timin·lifemin`（体素面积和） |
| `VolumeImage` / `overlap_window_size` | 72 / 6 | 一个块含 72 个时次（24 h），相邻块重叠 6 个时次（2 h），每块净推进 66 个时次 |
| `nbMaxCluster` | 100000 | 持久簇表 `p[]` 的大小（C 端另有硬编码 `NBMAX_LABEL_OBJECTS=1e6`） |

## 4. 核心算法：三维（x,y,t）层次化分割 —— "追踪"就是 3D 连通性

TOOCAN 不是"先逐帧识别、再做帧间对象匹配"的两步法。它直接在 (t,y,x) 体数据上做 3D 分割，
**时间追踪是 3D 连通性的副产物**：同一个标签贯穿对象的整个生命周期，合并/分裂由 3D 邻接与阈值层次自然给出。

连通性定义（`F_operator10`）：11 点算子 = 自身 + 8 个空间邻居 + 前/后各 1 个时间邻居（共 2 个时间邻居，
**没有**对角时间邻居）。Numba 的 `spread_labels_fast` 用等价的 10 邻域（不含自身）。

### 4.1 阈值循环（`detect_and_spread.py`）

```
labels ← 重叠区继承的标签（首块全 0）
for thr in 190,192,…,236:                    # 24 次
    nxt = min(thr+2, 235)
    (1) labels ← SPREAD(labels, mask = BT <  thr , Δ = 1)       # 已有对象先向暖侧长大
    (2) labels,N ← LABEL_REGION(thr)                              # 在 {0<BT≤thr} 中检测新种子
    (3) labels ← SPREAD(labels, mask = BT ≤ nxt , Δ = 1)         # 新种子长到下一个阈值
labels ← SPREAD(labels, mask = BT ≤ 235, Δ = 10)                  # 收尾
```

### 4.2 SPREAD（`spreading.py: spread_labels_fast`，Numba）

数学描述：令 `q(v)=trunc(100·BT(v))`（int16），`tb(v)=q(v)/100`。多源 FIFO 广度优先：初始队列 = 所有已标记体素（光栅序），
弹出 `c`，按固定邻居顺序 `(+t,−t,x−1,x+1,(y+1:x−1,x,x+1),(y−1:x−1,x,x+1))` 检查邻居 `n`：
若 `label(n)=0 ∧ mask(n)=1 ∧ tb(n)+Δ ≥ tb(c)` 则 `label(n)←label(c)` 并入队。

> **并行化的关键事实**：当一个体素同时可被多个标签到达时，归属由**队列先后顺序**决定（先到先得）。
> 因此该算法**不是**"各标签独立的形态学膨胀"，其结果依赖遍历顺序；任何空间/时间分块都可能在
> 竞争边界上改变像元归属，除非严格复现队列顺序（见 `TOOCAN_parallel_design.md`）。

### 4.3 LABEL_REGION（`label.c: Label_region`）

输入：`imseg`（冷体素 `0<BT≤thr` 的掩膜）、升序线性索引表 `indice_CloudyPix`、持久表 `p[]`（已识别 MCS，`flagFIX=1`）。

1. **单次升序扫描**（按线性索引顺序，非连通域标记算法，有顺序依赖）。对每个冷体素 `i`：
   * 跳过域边界像元（x=0, x=X−1, y=0, y=Y−1）。
   * 构造 11 点算子；`Nb` = 其中已有标签者个数。
   * "孤立像元"检验：算子项 1..6（x±1、(x−1,y+1)、y+1、(x+1,y+1)、(x−1,y−1)）全不是冷体素 → 跳过
     （**不检查** y−1、(x+1,y−1) 及两个时间邻居——源码的这一特征被忠实保留）。
   * `Nb==0` → 新标签 `++number`；否则取算子内按出现顺序的去重标签集 `U`（`uniq`）。
   * `|U|≤1` 且该标签未固定（`flagFIX≤0`）→ 把标签赋给算子内所有未标记冷体素。
   * `|U|>1` → 同样赋标签（取 `U[0]`），并在各标签的 `labelVoisin`（容量 1000，满则静默丢弃）里登记**邻接关系**。
2. **合并**：对邻接图中互相接触、且都不是"已固定 MCS"的标签做 DFS 重标记（`Relabel`，递归），根=下标最小者；
   若连通分量接触到已固定 MCS，则根的 `flagRelabel=1`，并记下 DFS 中**最后遇到**的已固定标签
   `labelMCS_alreadyidentified`。
3. **累计面积**：对冷体素表升序遍历，`seed_area_perFrame[z] += 像元面积`、`seed_area += 像元面积`。
   **数值细节**：这些是 `int` 变量，`int = int + float`，每次相加后**截断为整数**（且大于 2²⁴ 后 float 精度损失）。
4. **筛选**：寿命 `slotEnd−slotBegin+1 ≥ lifemin`；逐帧计"连续满足面积≥minAreaSeed 的时次数"（不满足且尚未达标则清零）；
   且 `seed_area ≥ minAreaSeed·minLifetime`。
5. **重编号**：通过者依次获得 `NSEEDS+1, NSEEDS+2, …` 并写入 `p[]`（`flagFIX=1`）；未通过者：若曾接触已固定 MCS 则并入该 MCS
   （`labelMCS_alreadyidentified`），否则标签置 0。

### 4.4 块间衔接（`run_toocan.py`）

块 k+1 的前 6 个时次 = 块 k 最后 6 个时次的**标签结果**（不是重新分割）。这些标签在块 k+1 内作为"已固定 MCS"
（`flagFIX=1`）存在，并在 `thr=190` 的第一次 SPREAD 中充当种子继续向新时次生长。`clean_clusters` 把持久表压缩为
仅含重叠区出现的标签，`labelMin = min(标签)−1`。全局计数器 `nb_ConvSeeds` 单调递增，保证标签全程唯一。
输出时，重叠时次以**新块的结果覆盖旧块**。

## 5. 输出

* **掩膜**（`save_labels_daily`）：`ToocanCloudMask_YYYYMMDD.nc`，`DCS_number(time,y,x)` int32，背景 −998，
  其余为 DCS 标签；2D `latitude/longitude`；zlib 压缩。
* **FileTracking**（`run_write_filetracking`）：对每个标签、每个时次计算质心、面积、冷云面积（≤235/220/210/200 K）、Tb 最小值/90 分位/均值、
  等效椭圆（235/220 K）→ `INT_*`（生命周期汇总）与 `LC_*`（生命周期曲线）变量，写入单个 NetCDF。

## 6. 哪些步骤可以独立、哪些必须连续（对并行设计的结论）

| 阶段 | 依赖 | 能否并行 |
|---|---|---|
| 读取 + OLR→BT 转换 | 仅依赖输入 | **完全独立**（逐时次） |
| 阈值循环内部（掩膜、冷体素表） | 仅依赖 BT | 数据并行 |
| SPREAD / LABEL_REGION | 依赖标签状态；先到先得 | **串行语义**（可用等价的层同步并行，见设计文档） |
| **块 k → 块 k+1** | 重叠区标签 + `nb_ConvSeeds` | **严格串行链**（全年约 400 块） |
| 掩膜 → FileTracking 统计 | 仅依赖掩膜+BT，逐时次独立 | **完全独立**（逐时次）；仅最后按标签聚合 |
| 不同实验/不同年份 | 无共享状态 | 完全独立（但跨年边界需上下文，见设计文档） |

## 7. 并行机制检查

* 无 OpenMP / MPI / pthread；`Label_region` 与 `spread_labels_fast` 都是单线程。
* Numba 函数未使用 `parallel=True`、`prange`，且每次调用都 JIT（未 `cache=True`）。
* `run_toocan.py` 把 `OMP_NUM_THREADS=1` 等环境变量在 SLURM 脚本中显式置 1。
* 现有"并行"仅是**作业层面**的：`run_monthly_segment.py`（按月/10 天分段）+ `stitch_segment_tracks.py`，
  属于近似（段间标签不一致，需事后拼接），不等价于连续运行。

## 8. 源码中值得注意的特征（影响重写时的等价性；均已被逐位复现）

1. `range(190, 235+2, 2)` 含 236 K，`threshold_next` 被截到 235。
2. 孤立像元检验只查 6 个空间邻居。
3. 域边界像元不参与检测，仅能经 SPREAD 获得标签。
4. 面积累计是"`int = int + float` 逐次截断"。
5. Numba 里 `(volume_bt*100).astype(np.int16)`：对 float32 数组乘 Python 整数，Numba 做的是 **float32 乘法**（`validation/test_quantization.py` 证明；float64 乘法在约 1% 的 0.01 K 边界值上不同）。BT>327.67 K 会溢出，但这些像元从不进入掩膜，无影响。
6. `Label_region` 每次调用分配并初始化 100 万个 Blob（每个含 1000 整数邻居表）→ 约 4.3 GB 写入（性能问题，非科学问题）。
7. `clean_clusters` 在 `labels_present` 为空时保留旧表和旧 `labelMin`（该分支行为未定义风险，实际数据中重叠区总有标签）。
8. 若 `n-labelMin-1 ≥ nbMaxCluster` 会越界写 `p[]`（C 端无检查）；新实现把它当作错误返回而不是静默越界。

9. `datetime.timestamp()`（FileTracking 的 `INT_UTC_*`、`LC_UTC`、`time`）把无时区日期当作**宿主机本地时间**，登录节点（CET）与计算节点（UTC）相差 3600 s；本实现按 UTC 计算。
10. 椭圆参数（`compute_equivalent_ellipse`）被计算并存入 `tracks`，但 `build_LC_variables` 从不使用它们——输出文件里椭圆变量恒为 −999。
11. `tracking_DCS.py` 中被注释掉的旧读取路径与 `creat_FileTracking.py`（611 行）是未被入口调用的旧版本；实际入口是 `run_write_filetracking.py` → `tracking_DCS.track_month_from_df`。

## 9. 参数文件格式注意
`parse_param_file` 只识别以 `;` 开头的整行注释，**不支持行内注释**（`minAreaSeed = 625 ; km2` 会被解析为字符串，进而在 `int()` 处出错）。示例配置因此把注释写在单独的行上。
