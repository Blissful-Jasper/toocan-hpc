# TOOCAN-HPC

面向超算的 TOOCAN 深对流系统（DCS）三维分割与目录（FileTracking）阶段的**逐位等价**高速重实现。
[TOOCAN](https://gitlab.in2p3.fr/thomas.fiolleau1/toocan-python)（Fiolleau & Roca 2013）对亮温（BT）做 3-D（经×纬×时间）
层次分割来追踪 DCS。本项目把耗时部分换成一个 OpenMP 的 C 库，外层是轻量 Python 流水线；输出标签与原始程序**逐体素完全相同**。

| | 原始 `toocan-python` | TOOCAN-HPC |
|---|---|---|
| 分割核心 | Python + `label.c`(ctypes) + Numba，单核 | 单个 C 库，OpenMP |
| 一个块（72 时次，601×3600） | ≈130 s | 2.1 s（16 线程）|
| 整年（400 块） | ≈20 h | 单节点 **19 分钟** |
| 目录阶段，全球 1 天 | 1998 s | 31 s（1 进程）/ 11.7 s（32 进程）|
| 中断后 | 整段重跑 | 从最后完成的一天续跑 |

验证与性能数据见 [docs/](docs/)。English: [README.md](README.md)。

## 1. 安装与编译

依赖：Linux、支持 OpenMP 的 `gcc`、Python ≥ 3.9（`numpy netCDF4 pandas xarray`；仅 MPI 目录阶段需要 `mpi4py`）。**不需要** Numba/Dask/CuPy。

```bash
git clone https://github.com/Blissful-Jasper/toocan-hpc.git && cd toocan-hpc
conda env create -f environment.yml && conda activate toocan-hpc     # 或 pip install -r requirements.txt
bash src/build.sh                                                     # 生成 src/libtoocan_core.so
python tests/test_smoke.py                                            # 约 30 秒，不需要任何数据
```
`build.sh` 使用 `-O3 -fopenmp -ffp-contract=off`，**不加** `-ffast-math`；改动这些选项会破坏逐位等价。换编译器：`CC=icc bash src/build.sh`。
在作业运行期间重新编译是安全的（原子重命名）。

## 2. 用合成数据试跑（1 分钟）

```bash
PY=python bash examples/demo.sh
```
脚本会编译、在 `demo_run/data` 生成 3 天假 OLR 和导航文件、跑分割和目录并打印摘要。它就是下面"用自己的数据"的最小教程。

## 3. 用你自己的数据

### 3.1 准备输入

需要两样东西。

**(a) OLR（或 BT）文件**：每天一个 NetCDF，放在任意目录树下（`path_ir` 会递归查找 `*.nc`）：

| 要求 | 说明 |
|---|---|
| 文件名 | 含 `YYYYMMDD`（用来按起止日期挑文件）|
| 变量 | `rlut`（或 `LWNTA`），**OLR，单位 W m⁻²**，维度 `(time, lat, lon)`；程序按原版公式转成 BT |
| 时间 | CF `time` 轴；间隔固定为 `temporalresolution` 分钟；时段内不能有缺时次 |
| 网格 | 规则经纬网，所有文件相同 |

**(b) 导航文件**：纬度、经度和**每个像元面积（km²）**（`minAreaSeed` 以 km² 计）：
```bash
python tools/make_navigation.py --from-file my_olr_day.nc --out navigation.nc     # 网格取自你的数据
# 或 python tools/make_navigation.py --lat -30 30 --lon 0 359.9 --res 0.1 --out navigation.nc
```

### 3.2 配置
```bash
cp config/geo_template.dat config/geo.dat     # 修改 REGION、temporalresolution、path_ir、file_navigation
$EDITOR config/toocan.dat                     # 科学参数；默认值 = 原版 TOOCAN v2.08 C5 配置
```
除非清楚后果，不要改 `VolumeImage = 72`、`overlap_window_size = 6` 和 BT 阈值（会影响标签）。`.dat` 里的注释必须独占一行（以 `;` 开头）。

### 3.3 分割，然后生成目录
```bash
export TZ=UTC HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1

# 1) 分割：整个时段一条连续链（起止含两端，然后 lonmin lonmax latmin latmax）
python src/run_toocan_hpc.py config/toocan.dat config/geo.dat 2000-01-01T00:00 2000-12-31T23:40  0 359.9 -30 30 \
       --out outputs/exp1 --threads 24 --readers 12 --writers 3

# 2) 目录：每行一个（DCS，时次），按天并行
python src/run_catalog_hpc.py config/toocan.dat config/geo.dat 2000-01-01T00:00 2000-01-31T23:40  0 359.9 -30 30 \
       --masks outputs/exp1 --out outputs/exp1/catalog --nproc 32
```
| 选项 | 含义 |
|---|---|
| `--threads N` | C 核心的 OpenMP 线程数（>32 收益很小）|
| `--readers N` / `--writers N` | 读入解码 / 写出压缩进程数 |
| `--fast-level {0,1,2}` | 0 逐行移植，1 受限扫描，2（默认）+并行 BFS；三者标签相同 |
| `--no-resume` | 忽略已有输出，从头计算 |
| `--ellipse`（目录）| 另算等效椭圆（原版输出里没有）|

**续跑**：崩溃/超时后原样再运行同一条命令即可，已完成的日文件被跳过，最多重算 2 块。两次运行之间不要改参数和时段。

### 3.4 SLURM（以一年为例）
```bash
mkdir -p logs                                  # 以下均在仓库根目录执行
export PY=$(which python) GEO=$PWD/config/geo.dat     # 可选：CFG LON0 LON1 LAT0 LAT1 OUTROOT THREADS
slurm/submit_pipeline.sh exp1 2000-01-01T00:00 2000-12-31T23:40 -A <账户> -p <分区>
```
先提交分割链作业，成功后再提交 12 个任务的目录数组。各脚本变量见 [slurm/](slurm/) 文件头注释。账户、分区等站点选项放在命令行；
不要多申请内存——对按 CPU 分配内存的分区，`--mem` 会悄悄放大 CPU 数。

### 3.5 输出
* 掩膜：`<out>/toocan_<version>/<REGION>/<YYYY>/<YYYY_MM_DD>/ToocanCloudMask_<YYYYMMDD>.nc`，`DCS_number(time,y,x)` int32；背景 −998，DCS 标签 ≥1 且全程唯一；`time` 为 1970 起算秒。
* 目录 `catalog_compact-<REGION>-<起>-<止>.nc`：每行一个（DCS，时次）：`obs_label, obs_epoch, obs_area, obs_lon_cm, obs_lat_cm, obs_tbmin, obs_tb90, obs_avg235/208/200, obs_pix_*, obs_km2_{235,220,210,200}`。
* `TOOCAN-<REGION>-<起>-<止>.nc`：原版目录格式（`INT_*`、`LC_*`）；DCS×时间矩阵超过 `dense_limit`（2×10⁸）时跳过。
* `<out>/.state/`：续跑状态与每块计时。

`python examples/inspect_output.py <out>` 打印输出摘要。

## 4. 与原版对比验证
```bash
python tests/test_smoke.py                                          # 自包含：各 fast-level / 线程数 / 续跑结果一致
TOOCAN_ORIG_ROOT=/path/to/toocan-python python tests/test_vs_original.py --seeds 3   # 合成场：核心 vs 原版
python validation/compare_masks.py   REF_ROOT NEW_ROOT              # 自己的两次运行：标签逐位比较
python validation/compare_objects.py REF_ROOT NEW_ROOT              # 对象级统计
python validation/compare_catalog.py REF.nc NEW.nc                  # 逐个 INT_/LC_ 变量
```

## 5. 与原版的差异（均为修正）
1. **时区**：原版 `INT_UTC_*`/`LC_UTC` 用 `datetime.timestamp()`，依赖主机时区；本项目固定 UTC（设 `TZ=UTC`），与原版在 UTC 主机上一致。
2. 簇数超过 `nbMaxCluster` 时原版越界写内存，本项目报错退出。
3. 每个日文件只打开一次（原版每时次打开一次）。

## 常见问题
* `KeyError: 'minAreaSeed'`：`.dat` 里写了行内注释，改成独占一行。
* `Missing IR slot inside non-terminal chunk`：输入时间序列有缺口（仅最后一块可以不满）。
* `tc_run_chunk error -7`：簇数超过 `nbMaxCluster`，增大它并检查重叠区。
* 标签与原版不同：检查编译选项、`--quant-mode`（保持 1）、`VolumeImage`/`overlap_window_size`。
* `libtoocan_core.so not found`：先执行 `bash src/build.sh`。

## 许可与引用
GPL-3.0（与原版 TOOCAN 相同，见 [LICENSE](LICENSE)、[NOTICE](NOTICE)）。使用结果请引用 Fiolleau & Roca (2013, IEEE TGRS, 10.1109/TGRS.2012.2227762) 与 (2024, ESSD, 10.5194/essd-16-4021-2024)。
