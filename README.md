## UAV 辅助测距定位时变距离扩大攻击仿真


## 1. 项目内容

本项目为无人机辅助无线传感器网络测距定位与攻击仿真代码，围绕无 GPS 待定位节点的坐标估计问题，构建无人机轨迹生成、多时刻测距、时变距离扩大攻击、定位求解和结果分析等模块。代码实现了 L2/L1 非线性定位、主动机动和参数配置等功能，支持调整轨迹形状、测距噪声、攻击启动时刻和偏差约束等参数。项目可用于理解 UAV 辅助定位流程，验证不同测距条件下的定位算法表现，并作为相关仿真教学、算法测试和二次开发的基础代码。
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

## 3. 绘图

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

##  如何运行

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

## Statement

项目名称（Project Name）：UAV 辅助测距定位时变距离扩大攻击仿真

项目作者（Author）：Jinwen He, Zhen Chen

作者单位（Affiliation）：暨南大学网络空间安全学院（College of Cyber Security, Jinan University）