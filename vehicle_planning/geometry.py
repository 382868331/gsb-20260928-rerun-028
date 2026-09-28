"""Small dependency-free geometry helpers."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: Point, b: Point, p: Point) -> bool:
    return (
        abs(_cross(a, b, p)) <= 1e-9
        and min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    c1 = _cross(a, b, c)
    c2 = _cross(a, b, d)
    c3 = _cross(c, d, a)
    c4 = _cross(c, d, b)
    if abs(c1) <= 1e-9 and on_segment(a, b, c):
        return True
    if abs(c2) <= 1e-9 and on_segment(a, b, d):
        return True
    if abs(c3) <= 1e-9 and on_segment(c, d, a):
        return True
    if abs(c4) <= 1e-9 and on_segment(c, d, b):
        return True
    return ((c1 > 0.0) != (c2 > 0.0)) and ((c3 > 0.0) != (c4 > 0.0))


def point_in_polygon(p: Point, polygon: Polygon) -> bool:
    x, y = p
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if on_segment((x1, y1), (x2, y2), p):
            return True
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_cross > x:
                inside = not inside
    return inside


def polygons_intersect(a: Polygon, b: Polygon) -> bool:
    """True when two simple polygons touch or overlap."""
    if point_in_polygon(a[0], b) or point_in_polygon(b[0], a):
        return True
    for i in range(len(a)):
        a1, a2 = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            if segments_intersect(a1, a2, b[j], b[(j + 1) % len(b)]):
                return True
    return False


def transform_points(points: Iterable[Point], x: float, y: float, theta: float) -> list[Point]:
    c = math.cos(theta)
    s = math.sin(theta)
    return [((px * c - py * s + x), (px * s + py * c + y)) for px, py in points]


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_distance(a: float, b: float) -> float:
    return abs(normalize_angle(a - b))
