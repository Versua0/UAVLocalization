## UAV 辅助测距定位时变距离扩大攻击仿真（当前版本）

本仓库目前包含两条思路：

- 旧链路：`EKF + NIS`（针对二维位置观测注入偏移）
- 新链路（当前任务重点）：`无人机轨迹 + 多次测距 + 时变距离扩大攻击 + L2/L1 定位`

本文档重点说明新链路的数据流、绘图数据来源和默认实验结果。

## 1. 问题定义（当前实现）

目标是定位一个无 GPS 的待定位节点，方法是让无人机沿轨迹飞行，在多个时刻对该节点测距，得到一组标量测距值后用定位算法求解节点坐标。

攻击者能力建模（已实现）：

- 能观测无人机轨迹
- 能对测距链路施加“距离扩大”攻击（只能增加距离，不能缩短）
- 注入是时变的，并随无人机轨迹几何关系变化
- 攻击目标是将最终定位结果拉向一个错误位置（`fake_position`）

## 2. 当前数据流（端到端）

入口文件：`src/range_localization_simulation.py`

数据流如下：

1. 生成无人机轨迹
- 文件：`data/input_data.py`
- 函数：`generate_trajectory_data(cfg.trajectory)`
- 输出：
  - `t`：时间序列
  - `uav_positions`：无人机位置序列 `(N, 2)`

2. 生成真实测距与含噪测距
- 文件：`src/range_measurement.py`
- 函数：
  - `compute_true_ranges(uav_positions, true_xy)` -> `ranges_true`
  - `generate_noisy_ranges(...)` -> `ranges_clean`
- 模型：
  - `r_k = ||u_k - p_true|| + n_k`  

3. 生成时变距离扩大攻击并注入
- 文件：`src/range_attack_simulation.py`
- 类：`TrajectoryAwareRangeInflationAttack`
- 核心思想：
  - 先用假目标位置 `p_fake` 构造期望偏差
  - `desired_bias_k = ||u_k - p_fake|| - ||u_k - p_true||`
  - 再做非负约束（距离扩大攻击）：`bias_k = max(desired_bias_k, 0)`
  - 再施加时序约束（启动时刻、最大偏差、变化率限制、平滑）
- 输出（`attack_out`）：
  - `ranges_attacked`：攻击后测距
  - `desired_bias`：理论目标偏差
  - `bias`：实际应用偏差
  - `active_mask`：攻击激活掩码
  - `extra_delay_s`：等效额外时延（可映射 M1/M2 中继时延）

4. 用 L2 / L1 解算目标位置
- 文件：`src/localization_solver.py`
- 方法：
  - `solve_localization_l2(...)`：Gauss-Newton 非线性最小二乘
  - `solve_localization_l1(...)`：IRLS 近似 L1
- 输入：
  - `uav_positions` + `ranges_clean` / `ranges_attacked`
- 输出：
  - `position`：估计位置
  - `residuals`：每个测距样本残差
  - `objective_l2` / `objective_l1`

5. 汇总指标与结果字典
- 文件：`src/range_localization_simulation.py`
- 函数：`run_range_localization_experiment()`
- 输出 `result` 字典（供绘图和分析使用）

## 3. 绘图（图里每个子图的数据来源）

绘图文件：`src/plot_range_localization_results.py`

默认生成图：`results/range_localization_attack.png`

### 图 1（左上）：几何关系与定位结果

数据来源：

- 无人机轨迹：`result["uav_positions"]`
- 真实目标点：`result["target_true_xy"]`
- 假目标点：`result["target_fake_xy"]`
- 干净测距 L2 解：`result["estimates"]["clean_l2"]["position"]`
- 受攻击测距 L2 解：`result["estimates"]["attacked_l2"]["position"]`
- 受攻击测距 L1 解：`result["estimates"]["attacked_l1"]["position"]`

含义：

- 直观看攻击是否把估计点从真实位置拉向假目标位置

### 图 2（右上）：测距序列（真实 / 含噪 / 攻击后）

数据来源：

- 真实测距：`result["ranges_true"]`
- 含噪测距：`result["ranges_clean"]`
- 攻击后测距：`result["attack"]["ranges_attacked"]`
- 攻击启动时刻：`result["config"].attack.start_time`

含义：

- 观察攻击前后测距波形差异，以及攻击在何时开始生效

### 图 3（左下）：时变攻击偏差与等效时延

数据来源：

- 目标偏差（理论）：`result["attack"]["desired_bias"]`
- 实际偏差（受约束后）：`result["attack"]["bias"]`
- 等效额外时延（ns）：`result["attack"]["extra_delay_s"] * 1e9`

含义：

- 展示攻击如何随轨迹变化，以及约束（平滑/变化率）如何影响最终注入波形

### 图 4（右下）：残差幅值（Clean L2 / Attacked L2 / Attacked L1）

数据来源：

- `abs(result["estimates"]["clean_l2"]["residuals"])`
- `abs(result["estimates"]["attacked_l2"]["residuals"])`
- `abs(result["estimates"]["attacked_l1"]["residuals"])`

含义：

- 比较受攻击后解算的残差分布，以及 L1 与 L2 的差异

## 4. 当前默认实验结果（示例）

使用默认配置（`build_default_range_localization_config()`）运行得到的一个结果示例：

- `clean_l2_error`: `0.0400 m`
- `clean_l1_error`: `0.0400 m`
- `attacked_l2_error_to_true`: `32.2726 m`
- `attacked_l2_error_to_fake`: `6.9541 m`
- `attacked_l1_error_to_true`: `32.2726 m`
- `attacked_l1_error_to_fake`: `6.9541 m`
- `attack_bias_mean/max`: `11.9765 / 28.3835 m`
- `attack_delay_max`: `189.354 ns`
- `success_l2 / success_l1`: `True / True`

解释：

- 未攻击时，定位误差约 `4 cm`
- 攻击后，L2/L1 解都显著偏离真实位置（约 `32 m`）
- 同时更接近假目标位置（约 `7 m`），说明攻击成功扭曲了解算结果

说明：

- 以上数值来自当前默认参数与随机种子（`random_seed=42`）
- 修改轨迹、假目标位置、攻击约束或噪声后，结果会变化

## 5. 如何运行（当前链路）

### 运行仿真并保存图

```powershell
python src/range_localization_simulation.py
```

输出：

- 控制台打印实验指标摘要
- 图像保存到 `results/range_localization_attack.png`

### 只运行不保存图

```powershell
python -c "from src.range_localization_simulation import main; main(plot=True, show_plot=False, save_plot=False)"
```

### 运行新增测试

```powershell
python -m pytest -q -p no:cacheprovider tests\test_range_attack_simulation.py tests\test_localization_solver.py
```

## 6. 关键参数（可直接调）

文件：`config/range_parameters.py`

建议重点调整：

- `attack.fake_position_xy`：攻击希望把定位结果拉向哪里
- `attack.start_time`：攻击启动时刻
- `attack.max_bias_m`：最大距离扩大幅度
- `attack.max_slew_rate_mps`：偏差变化率上限（更贴近中继硬件能力）
- `attack.smooth_window`：偏差平滑窗口
- `measurement.range_noise_std`：测距噪声强度
- `trajectory.*`：无人机轨迹形状与覆盖范围

## 7. 后续建议（与你场景更贴近）

当前实现为“理想攻击者”版本（构造偏差时使用了真实目标位置），适合作为上界分析。

如果要进一步贴近 M1/M2 场景，建议下一步做：

- 攻击者仅知道目标粗估计（而不是真值）
- 显式建模 M1->M2 高速链路时延和抖动
- 在约束下优化 `bias_k`（最大化定位扭曲、最小化被检测风险）  
