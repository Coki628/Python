# -*- coding: utf-8 -*-

"""
・わりとすんなり自力AC
・それぞれのスライムで、まず動かすコストを考えずに何回動かすと最善かを取っておく。
・動かす操作自体は全スライム1度で済むので、上記最善を全スライム分足し合わせたものに動かす操作のコストを後から足す。
・これを各回について確認して、最低コストになるものが答え。
・計算量N^2=2000^2=400万、pythonTLE,pypyAC0.4秒。
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

N, X = MAP()
A = LIST()

# dp[i][k] := スライムiについて、魔法をk回(以内)使った時の最低コスト
dp = list2d(N, N, INF)
for i in range(N):
    k = 0
    for j in range(i, i-N, -1):
        j %= N
        dp[i][k] = min(dp[i][(k-1)%N], A[j])
        k += 1

ans = [0] * N
for i in range(N):
    for j in range(N):
        # 各回に全スライム分のコストを足し合わせる
        ans[j] += dp[i][j]
    # 魔法使用分のコストはまとめて追加
    ans[i] += i * X
print(min(ans))
