# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc159/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2020/03/22/234317
・時間外に解けないか考えたけど、自力ならず。。
・部分和DP応用
・区間をどう取るかの情報を持たせるために、区間に関する3種類の状態を添字で管理する。
・計算量は3000^2=900万と状態に関する定数3*3/2程度。
・色々試行錯誤の末、TLE→AC。決め手は、DP配列の3次元を2次元にしたこと。
・そうだった、次元が深いと遅いんだよね。これでpypyAC1.1秒。
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

# dp0[i][j] := i番目の要素まで見て、総和がjで、使う区間が未確定の状態の通り数
dp0 = list2d(N+1, S+1, 0)
# dp1[i][j] := i番目の要素まで見て、総和がjで、使う区間がL確定済の状態の通り数
dp1 = list2d(N+1, S+1, 0)
# dp2[i][j] := i番目の要素まで見て、総和がjで、使う区間がLR確定済の状態の通り数
dp2 = list2d(N+1, S+1, 0)
dp0[0][0] = 1
for i, a in enumerate(A):
    for j in range(S+1):
        # aを総和に足さない遷移は全部OK
        dp0[i+1][j] += dp0[i][j]
        dp1[i+1][j] += dp0[i][j] + dp1[i][j]
        dp2[i+1][j] += dp0[i][j] + dp1[i][j] + dp2[i][j]
        # aを総和に足す遷移は、遷移元が状態2と遷移先が状態0は無理
        if j+a <= S:
            dp1[i+1][j+a] += dp0[i][j] + dp1[i][j]
            dp2[i+1][j+a] += dp0[i][j] + dp1[i][j]
        dp0[i+1][j] %= MOD
        dp1[i+1][j] %= MOD
        dp2[i+1][j] %= MOD
ans = dp2[N][S]
print(ans)
