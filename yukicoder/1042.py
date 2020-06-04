"""
・自力AC
・二分探索(小数)
・最近EPS使ったにぶたんでバグってTLEみたいなツイート見かけてたけど多分これのことかな。
　実際そうなって、回数決め打ちにしたら通った。
"""

import sys
from math import log2

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
EPS = 10 ** -10

def bisearch_max(mn, mx, func, times):
    """ 条件を満たす最大値を見つける二分探索(小数用) """

    ok = mn
    ng = mx
    for _ in range(times):
        mid = (ok+ng) / 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

p, q = MAP()

res = bisearch_max(1, INF, lambda n: n**2 <= p+q*n*log2(n), 1000)
print(res)
