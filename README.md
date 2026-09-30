# 车辆运动规划系统

低速非完整约束车辆的基础规划框架。车辆参数、场景、方案分别保存；搜索结果包含可回放的前进/倒车圆弧控制段，后续动态避障可在相同的运动、碰撞和轨迹接口上扩展。

## 模型

- 使用后轴自行车模型，状态为 `(x, y, theta)`，坐标点表示后轴中心。
- 运动基元是恒舵角圆弧，包含最大转角限制和前进、倒车两个挡位。
- 碰撞检查使用整车矩形轮廓；沿每条圆弧细分，不只检查端点。
- 目标同时约束位置和朝向。
- 规划结果状态明确区分 `path_found`、`no_path`、`search_limit_reached`。

## 动态障碍与等待

- 场景中的 `dynamic_obstacles` 是带 `(t, x, y, theta)` 关键帧的低速移动物体，位姿在关键帧间线性插值，区间外保持端点姿态。
- 静态几何规划完成后，`Scheduler` 在同一路径上叠加时间轴：车辆以恒定低速巡航，动作序列包含 `drive`、`wait`、`gear_switch`。
- `wait` 动作记录开始/结束时刻和等待原因（哪个临时障碍占用通道）；等待窗口与放行后的整车扫掠都按固定时间步逐点检查，不能只检查障碍最终位置。
- 每次前进/倒车方向切换都会插入固定时长的 `gear_switch` 过渡动作，车辆静止但时钟推进。
- 方案文件保存 `scene_signature`（覆盖场景几何与全部障碍时间表的哈希）。独立重放时逐时刻核对车身与障碍位置；时间表任何变化都会使旧方案失效，必须重新规划，旧的成功标记不会被沿用。
- 无法在最大等待时间内避开时返回 `schedule_failed`。

## 规划与回放

```bash
python -m vehicle_planning.cli plan \
  --vehicle data/vehicles/compact_car.json \
  --scene data/scenes/narrow_uturn.json \
  --output outputs/narrow_uturn.json

python -m vehicle_planning.cli replay \
  --vehicle data/vehicles/compact_car.json \
  --scene data/scenes/narrow_uturn.json \
  --plan outputs/narrow_uturn.json \
  --svg outputs/narrow_uturn.svg
```

同样替换为 `data/scenes/reverse_into_bay.json` 可回放倒入目标区样例。蓝色轨迹表示前进，红色表示倒车。

含临时横穿物体的场景会自动生成带等待的时间方案，也可显式加 `--timed`：

```bash
python -m vehicle_planning.cli plan \
  --vehicle data/vehicles/compact_car.json \
  --scene data/scenes/pedestrian_crossing.json \
  --output outputs/pedestrian_crossing.json

python -m vehicle_planning.cli replay \
  --vehicle data/vehicles/compact_car.json \
  --scene data/scenes/pedestrian_crossing.json \
  --plan outputs/pedestrian_crossing.json \
  --svg outputs/pedestrian_crossing.svg
```

规划输出会列出每个 `wait`/`gear_switch` 的起止时刻和等待原因；SVG 中橙色虚线框表示动态障碍关键帧位置。

## 障碍时间表变化后重开

方案与场景签名绑定。障碍时间表一旦变化（即使障碍只是移远、原几何路线其实仍走得通），旧方案不能单独“重新验证”，必须以“重开规划问题”的方式处理；原成功方案文件保持不变，失败结果不会覆盖旧的成功记录。

`python -m vehicle_planning.cli reopen` 保留原路线并对**新时间表**依次评估三种方案，路线变化与时间安排变化分别标记：

- `original`：原路线、原时间安排直接面对新时间表；给出是否仍可执行，以及逐时刻扫描出的具体冲突区间（起止时刻、持续时长、障碍名称、发生在 drive/换挡/等待哪个阶段）。
- `delayed`：先只调整出发/等待时间（整体平移时间轴并补一个初始等待动作），路线几何完全不变；找到最小可行延后量即采用。
- `replanned`：延后也无法避开时，在新时间表上重新搜索几何路径并重新排时，单独标记 `route_changed` 与 `timing_changed`。

```bash
python -m vehicle_planning.cli reopen \
  --vehicle data/vehicles/compact_car.json \
  --scene data/scenes/pedestrian_crossing_late.json \
  --plan outputs/pedestrian_crossing.json \
  --output outputs/reopen/report.json
```

报告会写入 `reopen_assessment_v1` JSON，并在同目录保存各可行方案（如 `report_delayed.json`、`report_replanned.json`）。每个重开方案都带**新时间表**的 `scene_signature`：对新时间表可独立回放成功，对旧时间表仍因签名不一致而失效；原方案只通过路径引用，绝不改写。覆盖障碍提前到达、延迟离开（需延后出发）以及完全避开路线（无几何冲突但仍必须重开）三类情况。

## 测试

```bash
python -m pytest -q
```
