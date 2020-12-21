"""
・自力AC
・解の公式
・昔作ったやつを貼る。で、昇順、重解を1つにまとめるとかよしなにする。
"""

import sys
from math import sqrt

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

# 2次方程式の解の公式
def dim2formula(a, b, c):
    from math import sqrt

    try:
        return [(-b+sqrt(pow(b, 2)-4*a*c))/(2*a), (-b-sqrt(pow(b, 2)-4*a*c))/(2*a)]
    # 解なしはNoneを返却
    except ValueError:
        return None, None

a, b, c = MAP()

ans = dim2formula(a, b, c)
if ans[0] is None:
    print('imaginary')
else:
    ans.sort()
    if abs(ans[0]-ans[1]) < EPS:
        print(ans[0])
    else:
        print(*ans)
