"""
・自力AC！ちょっと時間かけた。。
・二分探索
・たくさんベホマラーを使うと、ある回数以降は人が減って1人ずつベホイミした方が得をするラインがある。
　そのギリギリまでベホマラーして、残りはベホイミする。
・多分これで最善だと思ったけど、確証はなかったからね、通ってよかった。。
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

N, K, X, Y = MAP()
A = [a-1 for a in LIST()]

if X > Y:
    X = Y

# 残りをベホイミしても損をしない、ベホマラー回数の最小値
def check(m):
    cnt = 0
    for i in range(N):
        if A[i]-K*m > 0:
            cnt += 1
    return X*cnt <= Y

res = bisearch_min(-1, INF, check)

ans = Y * res
for i in range(N):
    A[i] = max(A[i]-K*res, 0)
for i in range(N):
    ans += ceil(A[i], K) * X
print(ans)
