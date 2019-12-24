# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1282/submission/67534529
・自力ならず。これは悔しい。。
・ソートしてDP、貰うDP
・ソートしておけば、前から貪欲に取れば最適になる。
・貰うDPで考えると、今回足す値が今までで1番高いものになって、ちょうどセット買いの条件と合わせられる。
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

for _ in range(INT()):
    N, P, K = MAP()
    A = [0] + LIST()

    A.sort()
    # dp[i] := i個購入する時の最小値
    dp = [INF] * (N+1)
    dp[0] = 0
    for i in range(1, N+1):
        # 単品買いの遷移
        dp[i] = min(dp[i], dp[i-1] + A[i])
        if i-K >= 0:
            # セット買いの遷移
            dp[i] = min(dp[i], dp[i-K] + A[i])

    for i, a in enumerate(dp):
        if a <= P:
            ans = i
    print(ans)
