"""
・なんとか自力AC。本日1番の難所だったｗ
・式変形で絶対いけるやつだと思ってゴニョゴニョしたけどうまくいかず、
　最終的ににぶたんして押し切った。
・ようは傾きが同じくらいになる場所を見つければいい。
　目標位置のxが小さいほど前の傾きは大きく、後ろの傾きは小さくなる。
　で、xは0があり得るけど制約からy<0なので分母をyにして比較。
　こうなるとさっき言った大小関係は逆になるので、不等号の向きは前>後ろにする。
・数学O(1)に失敗して20分くらい経ってしまった時はまじで精神が死にそうになった。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def bisearch_min(mn, mx, func, times):
    """ 条件を満たす最小値を見つける二分探索(小数用) """

    ok = mx
    ng = mn
    for _ in range(times):
        mid = (ok+ng) / 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

sx, sy, gx, gy = MAP()

def check(m):
    x1 = m - sx
    y1 = sy
    x2 = gx - m
    y2 = gy
    return abs(x1/y1) > abs(x2/y2)

res = bisearch_min(sx, gx, check, 100)
print(res)
