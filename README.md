# 车辆运动规划系统

低速非完整约束车辆的规划框架。车辆参数、场景、方案分别保存；方案包含带全局时钟的前进/倒车/等待/换挡动作，支持对低速移动障碍做时间维度避障与独立重放。

## 模型

- 使用后轴自行车模型，状态为 `(x, y, theta)`，坐标点表示后轴中心。
- 运动基元是恒舵角圆弧，包含最大转角限制和前进、倒车两个挡位。
- 静态碰撞与动态碰撞都检查整车矩形轮廓；每条圆弧按弧长细分，不只检查端点。
- 场景可包含 `dynamic_obstacles`：每个移动障碍有固定轮廓和关键帧时间表，位置/朝向按时间线性插值；在有效时间窗之外视为不存在，而不是把终点位置直接画到地图上。
- 目标同时约束位置和朝向。
- 规划结果状态明确区分 `path_found`、`no_path`、`search_limit_reached`。

## 时间、等待与换挡

- 方案的 `timeline` 按全局时钟列出三类控制动作：
  - `drive`：前进或倒车行驶段，时长等于距离除规划速度；
  - `wait`：整车原地停车等待，`reason` 记录被哪个移动障碍占用；
  - `shift`：前进/倒车（或倒车/前进）切换的静止过渡，默认耗时 2 秒。
- 停车位几何可达但通道被临时经过的物体占用时，搜索会自动插入 `wait`，等待期间和之后的行驶扫掠都逐时刻做碰撞检查。
- 每个方案保存 `timetable_signature`（动态障碍时间表的哈希）和 `makespan`。
- 重放时障碍位置始终从当前场景文件按时间戳重新计算；时间表变化后，即使方案文件仍带着旧的成功结果，也会重新验证并在报告中标记 `timetable_changed`，旧标记不会被直接沿用。

## 规划与回放

```bash
python -m vehicle_planning.cli plan \
  --vehicle data/vehicles/compact_car.json \
  --scene data/scenes/timed_crossing.json \
  --output outputs/timed_crossing.json \
  --speed 1.0 --gear-shift-time 2.0

python -m vehicle_planning.cli replay \
  --vehicle data/vehicles/compact_car.json \
  --scene data/scenes/timed_crossing.json \
  --plan outputs/timed_crossing.json \
  --svg outputs/timed_crossing.svg
```

`narrow_uturn` 与 `reverse_into_bay` 是仅含静态障碍的原有样例，不带时间线时仍按距离采样回放。`timed_crossing` 中绿色虚线为移动障碍轨迹与关键帧位置，橙色圆点表示等待位置，紫色圆点表示换挡位置。

## 测试

```bash
python -m pytest -q
```
