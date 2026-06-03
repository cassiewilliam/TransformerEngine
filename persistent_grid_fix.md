# SwiGLU grouped-GEMM persistent grid 修改方案

文件:`transformer_engine/common/gemm/cutlass_grouped_gemm_swiglu.cuh`
函数:`LaunchSwiGluGrouped(...)`(launch 段,约 L947–L989)

---

## 1. 问题定位

当前 launch 用**写死的 ×2 过订阅**来决定 persistent grid 大小:

```cpp
int cluster_size = int(cluster.x);                       // 2 (2-SM cluster) / 1 (1-SM)
int persistent_clusters = 2 * (sm_count_eff / cluster_size);   // ← 硬编码 ×2
```

以 B200(`sm_count=148`, `cluster=(2,1,1)`)为例:

```
persistent_clusters = 2 * (148/2) = 148 clusters
grid.x              = 148 * 2     = 296 CTAs  →  148 个 SM 上跑 296 个 CTA = 2 CTA/SM
```

**这不是正确性 bug**(grid-stride 循环 `for(tile=cluster_id; tile<total_tiles; tile+=num_clusters)` 对任意 `num_clusters` 都正确),**但 ×2 是一个性能/设计错招**:它假设每个 SM 能同时驻留 2 个 CTA,却**没有向硬件确认过**。

## 2. 为什么是问题

1. **这是个重 kernel**:`kStages=16`、大 smem、TMEM-heavy 的 Blackwell warp-specialized kernel,每个 SM 几乎肯定**只塞得下 1 个 CTA**。多发的 148 个 cluster 不会与第一波同时驻留,而是排成**第二波**。
2. **过订阅抵消了 persistent 的初衷**:persistent 化的目的是把 per-cluster 的 `TMEM alloc/free + cluster_sync` setup/teardown 摊薄到很多 tile 上;排第二波等于把这套 setup 又做一遍,还多一条 load-imbalance 尾巴。最好情况 no-op(被 occupancy=1 卡住),最坏情况比单波还慢。
3. **`12.5% → 25%` 这个论据站不住**:那是 Nsight 的 *achieved occupancy*(active warps / max warps)。warp-specialized persistent kernel(load/MMA/epilogue 分开,大量 warp 故意 idle)的 achieved occupancy 本来就低,**不是优化目标**。SM 忙不忙看 tensor core 有没有在做 MMA,不是 occupancy %。靠堆 block 数刷这个数字是认错了指标。

## 3. 参考实现(权威依据)

- **CUTLASS `StaticPersistentTileScheduler.get_grid_shape`**
  (`3rdparty/cutlass/python/CuTeDSL/cutlass/utils/static_persistent_tile_scheduler.py:173-176`):
  ```python
  num_ctas_per_wave      = max_active_clusters * num_ctas_per_cluster
  num_persistent_ctas    = min(num_ctas_in_problem, num_ctas_per_wave)   # 只发"一波"
  num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
  ```
  关键:**既不是 `sm/cluster_size`,也不是 `2*sm/cluster_size`**,而是 `min(problem_clusters, max_active_clusters)`,其中 `max_active_clusters` 是**实测**出来的(occupancy 是 1 就一波,是 2 自动两波,不用猜)。

- **CUTLASS C++ helper**
  `cutlass::KernelHardwareInfo::query_device_max_active_clusters(...)`
  (`3rdparty/cutlass/include/cutlass/kernel_hardware_info.h:85-104`)内部就是 `cudaOccupancyMaxActiveClusters`。
  ⚠️ 注意:这个 one-liner helper 构造 config 时**没带 dynamic smem**(`make_cluster_launch_config(cluster_dims, cluster_dims, {threads,1,1})`),对我们这种大 smem kernel 会**高估** occupancy。所以下面方案**自己构造带 `smem_size` 的 config**,不直接用 helper。

- **MegaMoE**:`shapes.h:39` 把 `num_sms` 直接注释为"参与的 SM 数(persistent grid 大小)",B200 填 148。即 **grid = num_sms 个 CTA = 单个 persistent wave**,无 ×2。

## 4. 修改方案(occupancy 驱动的单波 launch)

核心思路:用 `cudaOccupancyMaxActiveClusters`(带真实 `smem_size`)查出**设备上真正能同时驻留的 cluster 数**,直接用它作为 `persistent_clusters`;查询失败时退回 `sm_count/cluster_size`(单波、occupancy-1 假设,对齐 MegaMoE)。

> 实现要点:`cudaOccupancyMaxActiveClusters` 需要 `kernel_ptr` / `block` / `smem_size`,而这三个在原代码里定义在 grid 计算**之后**。因此需把它们(以及 `cudaFuncSetAttribute` 的 smem opt-in)**上移到查询之前**,确保查询看到的是真实的 smem 配置。

### Before(现状)

```cpp
  // PERSISTENT launch: ... We OVERSUBSCRIBE 2 clusters per SM-pair:
  //   persistent_clusters = 2 * (sm_count / cluster_size)
  int sm_count_eff = (sm_count > 0)
                         ? sm_count
                         : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  int cluster_size = int(cluster.x);
  int persistent_clusters = 2 * (sm_count_eff / cluster_size);          // ← ×2
  if (persistent_clusters < 1) persistent_clusters = 1;
  int num_clusters = persistent_clusters < total_tiles ? persistent_clusters : total_tiles;

  params.total_tiles = total_tiles;
  params.n_local_tiles = num_n_local_tiles;
  params.num_clusters = num_clusters;

  dim3 grid(num_clusters * cluster.x, 1, 1);
  dim3 block(Kernel::MaxThreadsPerBlock, 1, 1);
  int smem_size = Kernel::SharedStorageSize;
  void const* kernel_ptr = reinterpret_cast<void const*>(cutlass::device_kernel<Kernel>);

  if (smem_size >= (48 << 10)) {
    cudaError_t attr = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smem_size);
    if (attr != cudaSuccess) return attr;
  }
  cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  auto cfg = cutlass::ClusterLauncher::make_cluster_launch_config(grid, cluster, block, smem_size,
                                                                  stream);
  void* kernel_params[] = {&params};
  cudaError_t status = cudaLaunchKernelExC(&cfg.launch_config, kernel_ptr, kernel_params);
  return status;
```

### After(修改后)

```cpp
  // Kernel resource info (needed for the occupancy query below, so define it before the grid).
  dim3 block(Kernel::MaxThreadsPerBlock, 1, 1);
  int smem_size = Kernel::SharedStorageSize;
  void const* kernel_ptr = reinterpret_cast<void const*>(cutlass::device_kernel<Kernel>);

  if (smem_size >= (48 << 10)) {
    cudaError_t attr = cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smem_size);
    if (attr != cudaSuccess) return attr;
  }
  // Set both func attributes BEFORE the occupancy query so it sees the real smem opt-in + cluster.
  cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

  // PERSISTENT launch (occupancy-driven, SINGLE wave — mirrors CUTLASS StaticPersistentTileScheduler
  // and MegaMoE's grid = num_sms): launch exactly the number of clusters that ACTUALLY co-reside on
  // the device for THIS kernel, then grid-stride over total_tiles.  No guessing a multiple of the SM
  // count: a heavy (kStages, large-smem, TMEM) kernel typically packs only 1 CTA/SM, so over-launching
  // merely queues a second wave and re-pays the per-cluster TMEM alloc/free + cluster_sync setup the
  // persistent design exists to amortize.  cudaOccupancyMaxActiveClusters accounts for smem/reg/TMEM
  // AND cluster gang-scheduling.  cluster_size = ClusterShape.M (= cluster.x) is the CTAs per cluster.
  int cluster_size = int(cluster.x);
  int persistent_clusters = 0;
  {
    // Build the occupancy config WITH the real dynamic smem (the CUTLASS one-liner helper omits it and
    // would over-count).  grid = one cluster is the canonical "minimum valid grid";
    // cudaOccupancyMaxActiveClusters still returns the DEVICE-WIDE max active cluster count.
    auto occ_cfg = cutlass::ClusterLauncher::make_cluster_launch_config(cluster, cluster, block,
                                                                        smem_size, stream);
    int max_active_clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&max_active_clusters, kernel_ptr, &occ_cfg.launch_config) ==
        cudaSuccess) {
      persistent_clusters = max_active_clusters;
    }
  }
  if (persistent_clusters < 1) {
    // Fallback: one cluster per cluster_size SMs (single wave, occupancy-1 — matches MegaMoE).
    int sm_count_eff = (sm_count > 0)
                           ? sm_count
                           : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    persistent_clusters = sm_count_eff / cluster_size;
  }
  if (persistent_clusters < 1) persistent_clusters = 1;        // always launch at least one cluster
  // Capped at total_tiles so we never launch idle clusters (cluster_id >= total_tiles → zero iters).
  int num_clusters = persistent_clusters < total_tiles ? persistent_clusters : total_tiles;

  params.total_tiles = total_tiles;
  params.n_local_tiles = num_n_local_tiles;
  params.num_clusters = num_clusters;

  // Grid is 1-D in clusters: cluster.x CTAs per cluster → grid.x = num_clusters * cluster.x.
  dim3 grid(num_clusters * cluster.x, 1, 1);

  auto cfg = cutlass::ClusterLauncher::make_cluster_launch_config(grid, cluster, block, smem_size,
                                                                  stream);
  void* kernel_params[] = {&params};
  cudaError_t status = cudaLaunchKernelExC(&cfg.launch_config, kernel_ptr, kernel_params);
  return status;
```

## 5. 改动要点小结

| 项 | Before | After |
|---|---|---|
| persistent cluster 数 | `2 * (sm/cluster_size)`(猜) | `cudaOccupancyMaxActiveClusters`(实测),失败退回 `sm/cluster_size` |
| occupancy=1 时 grid | 296 CTA(两波) | 148 CTA(一波) |
| smem 是否纳入查询 | — | 是(自建带 `smem_size` 的 config,不用裸 helper) |
| 代码顺序 | grid 在前,`kernel_ptr/block/smem` 在后 | `kernel_ptr/block/smem` + smem opt-in 上移到查询之前 |
| 正确性 | OK | OK(grid-stride 循环不变) |

## 6. 验证建议

- **API 可用性**:`cudaOccupancyMaxActiveClusters` / `cudaLaunchConfig_t` 需 CUDA ≥ 11.8;Blackwell 必然满足。`ClusterLauncher` 当前代码已无条件使用,无需额外 guard。
- **数值不变**:输出与改前 bitwise 一致(只改 launch 维度,不改计算)。
- **性能**:在 B200 上对比改前/改后 grouped SwiGLU 的 kernel time;预期单波后 setup 开销下降,尾巴消失。顺带用 `ncu` 确认 `cudaOccupancyMaxActiveClusters` 返回值(occupancy 到底是 1 还是 2)以验证假设。
- **小 problem**:当 `total_tiles < persistent_clusters` 时仍由 `min(..., total_tiles)` 兜底,不会发空 cluster。
