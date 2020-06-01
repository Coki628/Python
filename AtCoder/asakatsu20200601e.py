"""
・終了約2分半でAC、、悔しいな。
・式変形、二分探索
・二分探索なことは覚えてたのに、方針思い出すのに時間かかってよくなかった。
・上限をINFにしてたらTLE。。ちゃんと考察してa*bなら最大10^12だなってやってAC。
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

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

N, K = MAP()
A = LIST()
B = LIST()
A.sort()
B.sort(reverse=1)

def check(m):
    C = [0] * N
    for i in range(N):
        C[i] = m // B[i]
    res = 0
    for i in range(N):
        res += max(A[i] - C[i], 0)
    return res <= K

res = bisearch_min(-1, 10**12+1, check)
print(res)
