"""
参考：https://www.youtube.com/watch?v=MXOSqPzsiqo&feature=youtu.be
・解説動画より
・座標を出して2点間距離でやる方針。ゲームプログラミングではこれはよく使うんだとか。
"""

import sys
from math import sin, cos, radians, hypot

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

a, b, h, m = MAP()

hm = h*60 + m

# 1周を1として針が今どこにあるか
heach = hm / 720
meach = m / 60
# *360で角度にする
hdig = heach * 360
mdig = meach * 360

# 角度と斜辺から、座標(底辺と高さ)を出す
hx = a * cos(radians(hdig))
hy = a * sin(radians(hdig))
mx = b * cos(radians(mdig))
my = b * sin(radians(mdig))

# 座標から2点間距離
ans = hypot(hx-mx, hy-my)
print(ans)
