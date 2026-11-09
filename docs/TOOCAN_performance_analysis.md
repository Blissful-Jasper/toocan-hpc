# 原始 TOOCAN 程序的性能瓶颈分析

> 注：文中的基准测试脚本与原始结果（`bench/`）保存在作者的工作目录，未随本仓库发布；所有数字均在 DKRZ Levante（AMD EPYC，256 线程/节点）上实测。

测量方法：`bench/profile_original.py` / `bench/profile_original_tracking.py` 在**不修改原始源码**的前提下，给驱动程序和
`detect_and_spread` 内部使用的模块级函数加计时包装（只做转调），然后用 `runpy` 原样执行 `toocan.run_toocan` /
`toocan.run_write_filetracking`。所有数字为 **[实测]**（Levante，`shared` 分区，单核，`bench/results/*.json`）。
性能对比、加速比与扩展性见 `TOOCAN_performance_report.md`。

## 1. 先回答"是不是只用了一个核"

是。`/usr/bin/time -v` 显示全域单块 92 % CPU（= 1 个核，其余是 I/O 等待）；源码里没有 OpenMP/MPI/pthread，
Numba 函数没有 `parallel=True`，SLURM 脚本里 `OMP_NUM_THREADS=1` 且 `--cpus-per-task=1`，作业申请 500 GB 内存只跑一个进程。
现有"并行"只是作业层面的月/十日分段（近似、需事后拼接）。

## 2. 分割阶段（`run_toocan`）逐阶段耗时

### 2.1 全域、一个块（1980-01-01，72 个时次，601×3600，8013 个 DCS）—— 总 149.6 s

| 阶段 | 耗时 | 占 `detect_and_spread` | 性质 |
|---|---|---|---|
| `Label_region`（C） | 35.8 s（24 次 ×1.49 s） | 28.4 % | CPU+内存：**每次调用分配并初始化 10⁶ 个 Blob（≈4.3 GB）** |
| `spread_labels_fast`（Numba） | 36.1 s（49 次，首次含 2.1 s JIT） | 28.6 % | CPU：每次扫描全体积建队列 + 单线程 BFS |
| `list_clusters`（`np.unique` 全体积，仅为了打印） | 20.0 s | 15.9 % | CPU：对 1.56 亿体素排序，**结果只用于 print** |
| `prepare_cloudy_indices`（`np.where` + `np.sort`） | 16.8 s | 13.3 % | CPU+内存：每个阈值重新生成冷体素索引并排序 |
| 其余（掩膜 `astype`、`copy`、差分、print） | 17.4 s | 13.8 % | 内存带宽 |
| **`detect_and_spread` 合计** | **126.1 s** | 100 % | |
| 读 72 个时次 `extract_volume` | 8.05 s | | I/O+解码（每时次 `xr.open_dataset`，0.109 s/时次） |
| 写日文件 `write_masks` | 4.22 s | | I/O+压缩 |
| 读导航 `crop_navigation_file` | 9.70 s（一次性） | | I/O（读整个 26 MB 文件并做 `np.ones` 外积） |
| 用户态 / 系统态 CPU | 83.1 s / **59.9 s** | | **系统态 40 %**：来自 Blob 的 4.3 GB × 24 次页错误与 calloc/free |
| 峰值内存 | 7.7 GiB | | |

### 2.2 小区域（301×601，3 天 = 4 个块）—— 总 182.3 s
`Label_region` 平均 **1.30 s/次**（96 次共 124.5 s）= `detect_and_spread` 总耗时（151.3 s）的 **82 %**。
区域缩小到 8 %，`Label_region` 耗时几乎不变 → 其时间与数据量无关，是纯固定开销（Blob 初始化）。

### 2.3 全域、3 天（1980-04-01..03，4 个块，21 837 个 DCS）—— 总 523.8 s

| 阶段 | 耗时 | 占总时间 |
|---|---|---|
| `detect_and_spread`（4 次） | 442.8 s | 84.5 % |
| 　`Label_region` | 132.2 s | 25.2 % |
| 　Numba SPREAD | 115.4 s | 22.0 % |
| 　`list_clusters` | 80.9 s | 15.4 % |
| 　`prepare_cloudy_indices` | 54.0 s | 10.3 % |
| 读 IR | 34.1 s | 6.5 % |
| 写掩膜 | 30.6 s | 5.8 % |
| `clean_clusters`（3 次） | 2.7 s | 0.5 % |
| 导航裁剪 | 9.7 s | 1.9 % |
| 峰值内存 | 8.7 GiB | |

> **生产环境佐证**：`logs/toocan_c5_1980_*` 记录了整年 1980 单作业：8 小时墙钟上限内完成 **160 个窗口（≈147 天）**，
> 即整年约 **20 小时**；作业摘要 MaxRSS = **88.7 GB**（对比 3 天测试的 8.7 GiB）—— 内存随块数增长（≈0.5 GB/块）。
> 同一作业的 `.out` 日志 **728 MB**（`list_clusters` 每个阈值打印每个标签）。

## 3. 瓶颈归类

### CPU 密集
1. 全体积重复扫描：SPREAD 每次调用扫描 1.56 亿体素建初始队列（49 次/块）；`prepare_cloudy_indices` 每阈值 `np.where`+`np.sort`；
   `list_clusters` 每阈值 `np.unique`。**这些都在重复处理"没有变化"的体素。**
2. `Label_region` 的 `Blob` 初始化（O(10⁶×1000) 的 memset）。
3. SPREAD 的 BFS 本身（单线程；队列长度 10⁶–10⁷）。

### I/O 密集
* 读：每时次 `xr.open_dataset` + `isel` + 裁剪（压缩 chunk 为 (1,72,3600)，读一个时次要解 9 个 chunk）；转换 OLR→BT 在同一线程里。
* 写：`save_labels_daily` 对每个块读回已有日文件、合并、再整文件重写（同一日在两个块里各写一次）。
* 标准输出：整年约 0.7 GB 文本。

### 内存瓶颈
* 每块：BT float32 + 标签 int32 + 辅助 uint8 掩膜 + int16 量化数组 + `volume_bt<thr` 的临时布尔/uint8（49 次分配 156 MB）+ `global_label_volume.copy()`。
* `clean_clusters` 每个块用 Python 循环给 100 000 个 Blob 各 `ctypes` 分配两块缓冲（≈0.4 GB）并逐个拷贝；旧表是否被及时释放未作验证（[推导]：88.7 GB / 160 块 ≈ 0.55 GB/块，与每块新建一个 ≈0.4 GB 的簇表同量级，提示存在随块数累积的内存增长）。
* Blob 表本身 4.3 GB × 每次 `Label_region`。

### 串行依赖
* 块内：阈值循环 24 次，每次 SPREAD→检测→SPREAD，严格串行。
* 块间：重叠区标签 + `nb_ConvSeeds`；**全年约 400 个块 = 一条链**（`TOOCAN_parallel_design.md` §1–2）。
* 对象生命周期追踪是 3D 连通性的一部分，不是后处理，因此"追踪"不能与分割分离。

### 通信与 GPU 传输（[推导]）
* 单进程，无通信。多节点方案的通信代价见设计文档 §2.1、§6（GPU 需把 624 MB 标签体往返 PCIe，约 25–50 ms/次）。

## 4. 目录阶段（`run_write_filetracking`）

小区域（301×601，216 个时次，**37 444 个"标签-时次"观测**）单核：**154.5 s**

| 阶段 | 耗时 | 说明 |
|---|---|---|
| `track_month_from_df` 总计 | 139.9 s | |
| 　椭圆 `compute_equivalent_ellipse`（74 888 次） | 45.5 s | **结果被计算但未写入任何输出**（`build_LC_variables` 里椭圆变量保持 −999 占位） |
| 　`compute_cold_surfaces`（37 444 次） | 19.8 s | 每标签 4 个阈值，各做一次整网格 `m & (bt<=thr)` |
| 　`read_ir_file`（216 次） | 19.7 s | 每时次 `xr.open_dataset` |
| 　`compute_temperature_stats` | 8.8 s | |
| 　循环自身（`mask==lab`、`np.nonzero`、`np.unique`） | ≈46 s | 每标签再扫描整网格 |
| `build_LC_variables` / `compute_INT_variables` / 写 NC | 0.48 / 0.39 / 1.10 s | Python 双重循环；LC 是 DCS×全局时间的稠密矩阵 |

**核心问题**：对每个时次的每个标签，原始代码对**整张网格**重复做约 10 次布尔运算（`mask == lab`、`m & (bt <= thr)`…），
代价 = O(标签数 × 网格大小)。小区域（18 万像元）平均 0.65 s/时次；**全域实测（1980-04-01 一天，72 个时次，75 819 个"标签-时次"观测，单核）：1998 s，即 27.8 s/时次**，
其中椭圆 935 s（47 %）、冷云面积 419 s、温度统计 95 s、读 IR 12.5 s，其余约 520 s 是循环自身的整网格 `mask==lab`/`np.nonzero`/`np.unique`。
整年 26 352 个时次按同样速率外推 ≈ **203 小时单核** [推导]。此阶段**逐时次完全独立**，是"令人尴尬的并行"。
另外稠密 LC 矩阵（DCS × 时间）随 DCS 数与时长二次增长，全年 ≈ 3×10⁶ 个 DCS（实测 ≈7 300 个/天）× 26 352 个时次 × 17 个变量 × 4–8 B ≈ **5–10 TB**，**不可行**，
因此必须按月分段或改用紧凑（逐观测）格式。

## 5. 结论：并行化的真正瓶颈

1. 最大的可消除成本是**重复/无用的工作**（Blob 初始化、`np.unique` 仅为打印、全体积重复扫描、整网格重复布尔运算、计算后丢弃的椭圆），
   不是"语言慢"。
2. 去掉这些之后，剩下的真正串行部分只有：前沿 BFS 与单遍贪心扫描（都是**仅对"有变化的体素"**工作，规模小两个数量级）和块间链。
3. 因此优先级：**先做精确的算法/结构优化（获得 ≈50× 且逐位等价），再做线程并行，最后才谈多节点**。
