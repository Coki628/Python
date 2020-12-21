"""
・自力AC
・前計算、累積和、二分探索
・累積和取って、にぶたんで1番近い条件を満たす位置を探す。
・でももっと簡単にやる方法もあった気はする。。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bisearch_min(mn, mx, func):
    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

def check(m):
    res1 = acc[0][m] - acc[0][i]
    res2 = acc[1][m] - acc[1][i]
    res3 = acc[2][m] - acc[2][i]
    return res1 and res2 and res3

for _ in range(INT()):
    S = input()
    N = len(S)

    acc = list2d(3, N+1, 0)
    for i, s in enumerate(S):
        if s == '1':
            acc[0][i+1] = 1
        elif s == '2':
            acc[1][i+1] = 1
        else:
            acc[2][i+1] = 1
    for s in range(3):
        acc[s] = list(accumulate(acc[s]))
    
    ans = INF
    for i in range(N):
        res = bisearch_min(i, N+1, check)
        if res != N+1:
            ans = min(ans, res - i)
    if ans == INF:
        print(0)
    else:
        print(ans)
