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

给定 `--earliest-departure` 与 `--latest-departure`（单位：相对原计划的延后秒数）后，报告会额外写入 `departure_windows`：**两个边界时刻首先探测**，随后内部按 `--departure-step` 等间隔探测（`boundaries` 中给出能否在该边界出发及占用障碍）。报告严格区分三级证据：

- 已测点：每个探测是某一出发时刻的点结论（`probes` 完整列出），输出零宽 `safe`/`blocked` 窗口且 `evidence_type="point"`，只代表该时刻本身；即使相邻采样点结论一致，也不外推到两点之间。
- 连续范围：正宽度的 `safe`/`blocked` 窗口必须携带独立连续覆盖证据（`evidence_type="continuous"`）。离散点探测无论多密，都不是连续覆盖证明；当前按步长扫描不会生成这种正宽度结论。
- 未核对范围：已测点之间的全部正宽度间隙、预算耗尽后的尾部、采样起点/步长变化后留下的间隙一律标记 `unverified`（`evidence_type="none"`），**不会**被写成安全，也不会被写成失败；只探测到两个可行端点时，中间整段保持未知。

窗口恰好铺满整个范围、互不重叠：已测点、有独立覆盖证明的连续范围、未核对范围三者严格区分。未被预算覆盖的边界时刻在 `boundaries` 中标记为 `probed=false`、`feasible=null`，不会被当作不可行。零宽范围（起止相同）只探测一次，给出零宽窗口。

分析会从已测可行点中选出最小可行出发时刻 `selected_departure`，按原路线生成带新签名的 `report_selected_window.json` 并全程回放核对（`selected_verified`）；`selection` 块记录它采用的场景、场景签名、路线指纹与核对结论，选择前可查看。没有已测可行点时 `selected_departure=null`，不生成可执行方案。

保存比较结果后可用 `--historical-report` 传入上一份报告继续搜索：场景签名与路线指纹都一致时视为**同一轮搜索追加预算**——旧探测原样保留、不重复计数（反复提交同一候选或中断后恢复都不会虚增核对数量），只有新位置消耗本轮预算，`newly_probed` 明确列出这次检查了哪些位置；缩小范围只做过滤，改变采样间隔（包括增大步长）不会改变旧点的证据性质，旧点之间的间隙仍保持 `unverified`；追加预算只会增加真正核对过的时刻并缩小未知间隙，不会把旧间隙自动升级为连续安全区间。签名或路线不一致时旧证据标记为 `needs_recheck`，全部重新探测，新旧场景的结论绝不拼成一段安全区间。上一份报告的 `selected_departure` 会被明确追踪：`prior_selection_status` 为 `kept`（旧选择仍在当前证据中核对可行）或 `invalidated`（场景/路线变化、该时刻出现冲突、或已不在当前核对集合内），并附 `prior_selection_note` 说明原因——旧选择失效一定显式说明，绝不静默沿用或静默丢弃。

## 测试

```bash
python -m pytest -q
```
