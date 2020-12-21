# -*- coding: utf-8 -*-

"""
・自力AC！
・更新順を工夫してDP
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

N, L = MAP()
A = [(a, i+1) for i, a in enumerate(LIST(N))]

# 長いつららから見られるようにソート
A.sort(reverse=1)
# dp[i] := i番目のつららが落ちる時間
dp = [0] * (N+2)
done = [0] * (N+2)
done[0] = done[-1] = 1
for a, i in A:
    # ここが落ちるまでの時間を追加
    dp[i] += L - a
    # 左右がまだ落ちていなければ、ここまでの経過時間を渡す
    if not done[i-1]:
        dp[i-1] = max(dp[i-1], dp[i])
    if not done[i+1]:
        dp[i+1] = max(dp[i+1], dp[i])
    done[i] = 1
ans = max(dp)
print(ans)
