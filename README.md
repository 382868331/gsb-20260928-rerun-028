# 车辆运动规划系统

这是一个纯 Python 的低速车运动规划原型，先打通“车辆参数 → 场景 → 可执行控制方案 → 独立轨迹回放与碰撞校验”的闭环。

## 已实现能力

- Ackermann 自行车模型，按轮距和最大前轮转角计算最小转弯半径。
- 搜索运动基元包含直行、左转、右转，并同时生成前进和倒车动作。
- 动作由连续时间运动学积分得到，轨迹包含直线段和圆弧段，不用节点间直线掩盖转弯不可行。
- 碰撞对象是整车矩形，而不是后轴中心点；回放时沿动作按固定弧长密集采样检查。
- 目标同时约束位置和朝向。
- 搜索结果明确区分：
  - solved：找到并生成控制序列；
  - no_path：搜索空间耗尽，未找到通路；
  - search_incomplete：扩展数或队列上限触发，尚不能证明无路。

## 数据组织

- configs/vehicles/compact_car.json：车辆尺寸和最大转角。
- configs/scenarios/*.json：起终点、朝向、矩形障碍、目标容差和边界。
- plans/*.plan.json：保存车辆/场景标识、状态、初始/目标位姿和可执行控制序列。
- trajectories/*.trajectory.json：由保存的方案重新积分得到的离散轨迹，仅用于回放和检查。

方案中不保存“节点连线”。每个 MotionCommand 都包含：

- steering_angle：前轮转角；
- velocity：带符号速度，负数表示倒车；
- duration：持续时间。

## 使用

    python -m vehicle_planner.cli plan \
      --vehicle configs/vehicles/compact_car.json \
      --scenario configs/scenarios/narrow_uturn.json \
      --output plans/narrow_uturn.plan.json \
      --heading-bins 6

    python -m vehicle_planner.cli replay \
      --vehicle configs/vehicles/compact_car.json \
      --scenario configs/scenarios/narrow_uturn.json \
      --plan plans/narrow_uturn.plan.json \
      --trajectory trajectories/narrow_uturn.trajectory.json

## 内置样例

- narrow_uturn：窄位掉头，三段极限转弯，包含倒车挡切换。
- reverse_into_bay：倒入两车之间的目标位姿。
- blocked_box：封闭障碍，用于验证 no_path 与求解超时分离。

运行全部测试：

    python -m unittest discover -s tests -v

## 动态避障扩展点

当前 vehicle_planner.collision 负责静态矩形障碍和边界检查。下一步加入动态障碍时，不需要修改运动学或方案格式，可将其扩展为接收时间的碰撞判定：

1. 在场景模型中增加带时间轨迹的动态障碍；
2. 在路径采样时根据累计时间查询动态障碍位姿；
3. 搜索状态中加入时间维度，或在静态可行路径上做时空速度规划；
4. 继续使用 replay_plan 对保存的控制序列做独立时空碰撞复核。
