# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc159/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2020/03/22/234317
・時間外に解けないか考えたけど、自力ならず。。
・部分和DP応用
・区間をどう取るかの情報を持たせるために、区間に関する3種類の状態を添字で管理する。
・計算量は3000^2=900万と状態に関する定数3*3/2程度、でもこのままだとpypyTLE。。
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
MOD = 998244353

N, S = MAP()
A = LIST()

# dp[i][j][k] := i番目の要素まで見て、総和がjで、使う区間がk=0,1,2(0:未確定, 1:L確定済, 2:LR確定済)の状態の通り数
dp = list3d(N+1, S+1, 3, 0)
dp[0][0][0] = 1
for i, a in enumerate(A):
    for j in range(S+1):
        for k in range(3):
            if dp[i][j][k] == 0:
                continue
            for l in range(k, 3):
                # aを総和に足さない遷移は全部OK
                dp[i+1][j][l] += dp[i][j][k]
                dp[i+1][j][l] %= MOD
                # aを総和に足す遷移は、遷移元が状態2と遷移先が状態0は無理
                if j+a <= S and k != 2 and l != 0:
                    dp[i+1][j+a][l] += dp[i][j][k]
                    dp[i+1][j+a][l] %= MOD
ans = dp[N][S][2]
print(ans)
