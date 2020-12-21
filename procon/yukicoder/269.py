# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/01/16/222843
・蟻本演習2-3-7、分割数
・考察の理解については自分ノート参照。
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

# 分割数：自然数NをK個の0以上の整数の和で表す場合の数
def partition(N, K):
    # dp[i][j] := jのi分割の総数
    dp = list2d(K+1, N+1, 0)
    dp[0][0] = 1
    for i in range(1, K+1):
        for j in range(N+1):
            dp[i][j] += dp[i-1][j]
            if j - i >= 0:
                dp[i][j] += dp[i][j-i]
            dp[i][j] %= MOD
    return dp[K][N]

# 等差数列の和：(初項a, 末項b, 項数c)
def get_sum(a, b, c):
    return (a+b) * c // 2

N, S, K = MAP()

# 各N人について固定で使わなければいけない数の合計
fixed = get_sum(K, K*(N-1), N-1)
# 上記がS以内なら実現可能
if fixed <= S:
    print(partition(S-fixed, N))
else:
    print(0)
