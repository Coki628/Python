# -*- coding: utf-8 -*-

"""
参考：https://rsk0315.hatenablog.com/entry/2020/05/02/181650
・括弧系、作ってみると割と普通のナップザック系2次元DP
・こういう状態と遷移、ちゃんと作れるようにならないとなぁ。。
・計算量3000^2=900万でpythonTLE,pypyAC0.2秒。
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

N = INT()
P = [1 if s == '(' else -1 for s in input()]
A = LIST()
B = LIST()

# dp[i][j] := i文字目まで見て、開いている括弧がj個ある時の最小コスト
dp = list2d(N+1, N+1, INF)
dp[0][0] = 0
for i, p in enumerate(P):
    for j in range(i+1):
        # 操作しない
        if j+p >= 0:
            dp[i+1][j+p] = min(dp[i+1][j+p], dp[i][j])
        # 変更
        if j-p >= 0:
            dp[i+1][j-p] = min(dp[i+1][j-p], dp[i][j] + A[i])
        # 削除
        dp[i+1][j] = min(dp[i+1][j], dp[i][j] + B[i])
ans = dp[N][0]
print(ans)
