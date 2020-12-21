# -*- coding: utf-8 -*-

"""
参考：https://rsk0315.hatenablog.com/entry/2019/12/29/051900
・自力ならず。。これは悔しい。式変形考えたけど正しい形を導けなかった。
・二分探索、式変形
・sum(B)/sum(A) >= x -> sum(B-A*x) >= 0 とする。
・sum()を含む方程式もちゃんと変形できるようにしないとな。。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bisearch_max(mn, mx, func, EPS):
    """ 条件を満たす最大値を見つける二分探索(小数用) """

    ok = mn
    ng = mx
    while ok+EPS < ng:
        mid = (ok+ng) / 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

# 合成後の値でxを達成可能か
def check(x):
    A2 = []
    for a, b in A:
        # 式変形：sum(B)/sum(A) >= x -> sum(B-A*x) >= 0
        A2.append(b - a*x)
    A2.sort(reverse=1)
    B2 = []
    for a, b in B:
        B2.append(b - a*x)
    B2.sort(reverse=1)
    # Aから5体
    AB = A2[:5]
    # Bの1番目がAの5番目より良ければ入れ替え
    if AB[4] < B2[0]:
        AB.pop()
        AB.append(B2[0])
    return sum(AB) >= 0

N, M = MAP()
A = []
for _ in range(N):
    a, b = MAP()
    A.append((a, b))
B = []
for _ in range(M):
    a, b = MAP()
    B.append((a, b))

ans = bisearch_max(0, 10**6, check, 10**-8)
print(ans)
