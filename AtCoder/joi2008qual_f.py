# -*- coding: utf-8 -*-

"""
参考：https://comprolog.netlify.com/post/joi/2009qual6/
・誤読で難しいこと考えてた。。N*Nが普通に1本の数列として見れる。
・素直に3次元DPするとMLEするので2次元使い回しで。
・計算量がM*N^2*S=2億9400万にもなるんだけど、C++はAC0.8秒。
・添字としては、何番目の位置まで見て、ではなく、
　何番目の値まで見て、を1次元目でindexにするとうまくいく。
・これは愚直なのでMLE想定。
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
MOD = 100000

N, M, S = MAP()

NN = N ** 2
# dp[a][j][k] := 値aまで見て、位置jまで確定して、総和がkの時の通り数
dp = list3d(M+1, NN+1, S+1, 0)
dp[0][0][0] = 1
for a in range(1, M+1):
    for j in range(NN+1):
        for k in range(S+1):
            dp[a][j][k] += dp[a-1][j][k]
            if j-1 >= 0 and k-a >= 0:
                dp[a][j][k] += dp[a-1][j-1][k-a]
            dp[a][j][k] %= MOD
ans = dp[M][NN][S]
print(ans)
