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
- 方案文件同时保存 `scene_signature`、`route_signature`（只覆盖起点/终点与几何路径）和 `timetable_signature`（只覆盖动态障碍时间表），并以 `variant` 标记 `original`/`delayed`/`replanned`。独立重放时逐时刻核对车身与障碍位置；时间表任何变化（即使障碍只是移远、完全避开路线）都会使旧方案失效，不能对旧路线单独“重新验证”，必须按新时间表重新规划，旧成功记录永不被沿用或改写。

## 时间表变更后的重开

`replan` 子命令对已保存方案做**只读评估**，原方案文件不会被改动。它同时比较三种方案，并明确区分“路线变化”和“时间安排变化”：

- `original`：同一路线、同一时间安排。报告在新时间表下是否仍可执行，并给出具体冲突时间区间（哪个障碍、跨越哪些 drive/wait/gear_switch 动作）。即使只读评估通过（障碍已完全避开），原成功记录也不改动。
- `delayed`：路线完全不变，先尝试整体延后出发时间，再退化为在同一几何路径上重排等待；`route_signature` 保持不变。
- `replanned`：仅当调整时间也无法避开时才改变行驶路径，针对新时间表搜索并排定全新路线。

```bash
python -m vehicle_planning.cli replan \
  --vehicle data/vehicles/compact_car.json \
  --scene data/scenes/pedestrian_crossing.json \
  --plan outputs/pedestrian_crossing.json \
  --report outputs/replan_report.json \
  --output outputs/pedestrian_crossing_delayed.json
```

`--report` 保存三方案对比（可行性、`route_changed`、`timing_changed`、冲突区间、推荐方案），`--output` 只保存推荐的 `delayed`/`replanned` 变体；原文件始终保持不变。加 `--no-route-replan` 可只评估时间调整。重开后每个变体仍带自己生成时的 `timetable_signature`：新变体能且只能在对应时间表下回放通过，旧变体在新时间表下仍被拒绝。障碍提前到达、延迟离开、完全避开路线三类情形都有测试覆盖。
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

## 测试

```bash
python -m pytest -q
```
