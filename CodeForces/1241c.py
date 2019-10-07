# -*- coding: utf-8 -*-

"""
・自力AC、二分探索
・最適な位置はにぶたんで出せる、の方針に割とすぐ気づけたのは良かった。
・クエリ20万だけどNの合計も20万のやつで、普通に20万とみなしていいか心配だったけど、
　O(NlogN)？がpython1.3秒、pypy0.6秒でAC。
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
INF = float('inf')
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

def calc(m):
    A2 = A[:m]
    A2.sort(reverse=True)
    ans = 0
    for i in range(len(A2)):
        ans += P[i] * A2[i]
    return ans >= K

for i in range(INT()):
    N = INT()
    P = [p//100 for p in LIST()]
    x, a = MAP()
    y, b = MAP()
    K = INT()

    P.sort(reverse=True)
    A = [0] * N
    for i in range(1, N+1):
        if i % a == 0:
            A[i-1] += x
        if i % b == 0:
            A[i-1] += y
    res = bisearch_min(0, N+1, calc)
    if res == N+1:
        print(-1)
    else:
        print(res)
