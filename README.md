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

方案与场景签名绑定。障碍时间表一旦变化（即使障碍只是移远、原几何路线其实仍走得通），旧方案不能靠签名校验“重新验证”，必须以“重开规划问题”的方式处理；重开时原路线会**独立面对新时间表**逐时刻核对——零冲突即判可行，与“延后零秒、时间安排未变”的方案结论一致；原成功方案文件保持不变，失败结果不会覆盖旧的成功记录。

`python -m vehicle_planning.cli reopen` 保留原路线并对**新时间表**依次评估三种方案，路线变化与时间安排变化分别标记：

- `original`：原路线、原时间安排直接面对新时间表（绕开签名校验做独立重放）；给出是否仍可执行，以及逐时刻扫描出的具体冲突区间（起止时刻、持续时长、障碍名称、发生在 drive/换挡/等待哪个阶段）。零冲突时与 `delayed` 的 0 秒延后判定一致。
- `delayed`：先只调整出发/等待时间（整体平移时间轴并补一个初始等待动作），路线几何完全不变；找到最小可行延后量即采用。
- `replanned`：延后也无法避开时，在新时间表上重新搜索几何路径并重新排时，单独标记 `route_changed` 与 `timing_changed`。

```bash
python -m vehicle_planning.cli reopen \
  --vehicle data/vehicles/compact_car.json \
  --scene data/scenes/pedestrian_crossing_late.json \
  --plan outputs/pedestrian_crossing.json \
  --output outputs/reopen/report.json
```

报告会写入 `reopen_assessment_v1` JSON，并在同目录保存各可行方案（如 `report_original.json`、`report_delayed.json`、`report_selected_window.json`、`report_replanned.json`）。每个重开方案都带**新时间表**的 `scene_signature`：对新时间表可独立回放成功，对旧时间表仍因签名不一致而失效；原方案只通过路径引用，绝不改写。覆盖障碍提前到达、延迟离开（需延后出发）以及完全避开路线（无几何冲突但仍必须重开）三类情况。

### 出发时间安全区间

给定 `--earliest-departure` 与 `--latest-departure`（单位：相对原计划的延后秒数）后，报告会额外写入 `departure_windows`。边界和采样时刻是零宽的具体核对点：`checked_probes` 保存实际结论、核对采样数、场景签名、路线签名和核对编号；点与点之间写入开区间或半开区间形式的 `unverified`。预算耗尽只意味着证据不足，不会产生安全区间，也不会把未检查的最晚边界判成失败，未核对边界显示为 `feasible: null`。

- `safe`（`evidence_type: point`）：仅该具体出发时刻沿当前路线全程扫描无冲突，可在该时刻执行；
- `blocked`（`evidence_type: point`）：仅该具体出发时刻被具体障碍占用，记录障碍名称；
- `unverified`：尚未核对的开区间或半开区间，不会被写成安全或失败。

只有存在独立、明确的连续依据时才输出 `evidence_type: continuous` 的 `safe` 区间；报告会列出 `basis` 与 `boundary_conditions`，说明上下界、障碍关键帧出现/消失时刻以及保持段是否已经包含。仅核对两个端点或若干采样点不会推出中间时刻可行。零宽时间范围会得到一个零宽点证据；端点相接的占用区间按同一套闭/开边界语义处理。

分析只会从已核对可行点或有连续依据的区间选择 `selected_departure`，并在 `selected_evidence` 中给出采用的场景、路线和实际核对结论；`selection_status` 为 `provisional` 时允许保存方案，但不会生成可执行的 `report_selected_window.json`。

预算不足后可用 `--continue-report` 传入上一份报告继续同一轮搜索。`newly_checked_departures` 只列本轮新核对的位置；重复提交同一候选不会增加 `check_count` 或伪造新的独立核对。也可用 `--departure-candidate 3.2` 直接检查两个已核对点之间的新位置。缩小范围或改变采样间隔会保留同场景、同路线的旧证据；场景或路线版本变化时，旧证据标记为 `scene_changed` 或 `route_changed` 并隔离，绝不与新场景的结果拼接。

保存比较结果后再次更换障碍安排时，可用 `--historical-report` 传入上一份报告：旧时间表下的区间只作为历史结论标注——签名一致为 `same_timetable`（仍需按需复核），签名不同为 `needs_recheck`，当前窗口的安全/占用结论一律来自新时间表的新鲜探测，绝不沿用旧标记。

## 测试

```bash
python -m pytest -q
```
