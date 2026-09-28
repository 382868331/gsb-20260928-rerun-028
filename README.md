# 车辆运动规划系统

低速非完整约束车辆的基础规划框架。车辆参数、场景、方案分别保存；搜索结果包含可回放的前进/倒车圆弧控制段，后续动态避障可在相同的运动、碰撞和轨迹接口上扩展。

## 模型

- 使用后轴自行车模型，状态为 `(x, y, theta)`，坐标点表示后轴中心。
- 运动基元是恒舵角圆弧，包含最大转角限制和前进、倒车两个挡位。
- 碰撞检查使用整车矩形轮廓；沿每条圆弧细分，不只检查端点。
- 目标同时约束位置和朝向。
- 规划结果状态明确区分 `path_found`、`no_path`、`search_limit_reached`。

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

## 测试

```bash
python -m pytest -q
```
