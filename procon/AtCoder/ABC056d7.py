# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc070/editorial.pdf
・部分和のDPやるのは正しくて、あと一歩が必要だった。。
・単調性が成り立つので二分探索でどこまで必要か判定できる。
　これ気付けなかったー。単調じゃない気がしたんだけど、よく確認すべきだった。
・でもこれまだTLE。O(NKlogN)=2500万*約12=3億。。。無理だろこれ。
　実際、にぶたんをあんま疑わなかったのは、この計算量を感じたのもある。
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
    """ 条件を満たす最小値を見つける二分探索 """

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

N, K = MAP()
A = LIST()

def check(omit):
    # 部分和DP
    dp = [0] * K
    dp[0] = 1
    for i in range(1, N+1):
        for j in range(K-1, -1, -1):
            if i-1 != omit and j-A[i-1] >= 0:
                dp[j] |= dp[j-A[i-1]]
    # 今回の値A[omit]を足してKに到達できるような部分和があれば、これは必要
    for j in range(max(0, K-A[omit]), K):
        if dp[j]:
            return True
    # なければ不要
    return False

A.sort()
print(bisearch_min(-1, N, check))
