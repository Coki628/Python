# -*- coding: utf-8 -*-

from math import hypot

a, b, c = map(int, input().split())
x1, y1, x2, y2 = map(int, input().split())

# 係数aかbが0だと、道は垂直か水平なのでマンハッタンと変わらない
if a == 0 or b == 0:
    print(abs(x1 - x2) + abs(y1 - y2))
    exit()

# x, yについて解いて各交点を出す
# x = (by + c) / -a
# y = (ax + c) / -b
A_x = (b*y1 + c) / -a
A_y = (a*x1 + c) / -b
B_x = (b*y2 + c) / -a
B_y = (a*x2 + c) / -b

# 使える範囲内にある交点か調べる
points = []
if x1 <= A_x <= x2 or x1 >= A_x >= x2:
    points.append((A_x, y1))
if y1 <= A_y <= y2 or y1 >= A_y >= y2:
    points.append((x1, A_y))
if x1 <= B_x <= x2 or x1 >= B_x >= x2:
    points.append((B_x, y2))
if y1 <= B_y <= y2 or y1 >= B_y >= y2:
    points.append((x2, B_y))
# 重複削除(始点終点と交点が一致してる時対応)
points = list(set(points))

# 範囲内の交点が2つあったら
if len(points) == 2:
    # Aから垂直か水平位置に交点1があり、Bから垂直か水平位置に交点2がある
    if ((points[0][0] == x1 or points[0][1] == y1)
            and (points[1][0] == x2 or points[1][1] == y2)):
        # 通れる道でも向きによっては遠くなるので、マンハッタンと比較する
        print(min(hypot(x1 - points[0][0], y1 - points[0][1])
            + hypot(points[0][0] - points[1][0], points[0][1] - points[1][1])
            + hypot(points[1][0] - x2, points[1][1] - y2),
            abs(x1 - x2) + abs(y1 - y2)))
    # 交点1と2が逆
    elif ((points[1][0] == x1 or points[1][1] == y1)
            and (points[0][0] == x2 or points[0][1] == y2)):
        print(min(hypot(x1 - points[1][0], y1 - points[1][1])
            + hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
            + hypot(points[0][0] - x2, points[0][1] - y2),
            abs(x1 - x2) + abs(y1 - y2)))
    # 交点1と2が両方Aから垂直か水平位置にあるなど、道が使えない向きの場合
    else:
        print(abs(x1 - x2) + abs(y1 - y2))
# いい感じの範囲に交点がなければ普通にマンハッタン
else:
    print(abs(x1 - x2) + abs(y1 - y2))
