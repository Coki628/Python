"""
・自力AC
・解の公式、2点を通る直線の式
・前に使った計算式をそのまま流用できたからよかった。
・やっぱり数学問は難易度の割に時間かかる。これ茶diffか。。
"""

import sys

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

# 2点を通る直線の式の係数a,b
def calc(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = (y2-y1)/(x2-x1)
    b = -(x1*y2/(x2-x1))+x1*y1/(x2-x1)+y1
    return a, b

# 2次方程式の解の公式
def dim2formula(a, b, c):
    from math import sqrt

    try:
        return (-b+sqrt(pow(b, 2)-4*a*c))/(2*a), (-b-sqrt(pow(b, 2)-4*a*c))/(2*a)
    # 解なしはNoneを返却
    except ValueError:
        return None, None

a, b, c, d = MAP()

ac = a - c
bd = b - d
# 解の公式から交点のx座標が求まる
x1, x2 = dim2formula(2, ac, bd)
# 交点なし
if x1 == x2 == None:
    No()
    exit()
# 交点1つ
if x1 == x2:
    Yes()
    exit()
# xを元の式に代入してy座標を出す
y1 = x1**2 + a*x1 + b
y2 = -x2**2 + c*x2 + d
# 2点を通る直線の式
ans = calc((x1, y1), (x2, y2))
print(*map('{:.10f}'.format, ans))
