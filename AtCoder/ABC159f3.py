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
・さらに2つだけ持って再利用するやつ使って1次元にした。これでpypyAC0.9秒。
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

dp0 = [0] * (S+1)
dp1 = [0] * (S+1)
dp2 = [0] * (S+1)
nxt0 = [0] * (S+1)
nxt1 = [0] * (S+1)
nxt2 = [0] * (S+1)
dp0[0] = 1
for i, a in enumerate(A):
    for j in range(S+1):
        # aを総和に足さない遷移は全部OK
        nxt0[j] += dp0[j]
        nxt1[j] += dp0[j] + dp1[j]
        nxt2[j] += dp0[j] + dp1[j] + dp2[j]
        # aを総和に足す遷移は、遷移元が状態2と遷移先が状態0は無理
        if j+a <= S:
            nxt1[j+a] += dp0[j] + dp1[j]
            nxt2[j+a] += dp0[j] + dp1[j]
        nxt0[j] %= MOD
        nxt1[j] %= MOD
        nxt2[j] %= MOD
    dp0 = nxt0[:]
    dp1 = nxt1[:]
    dp2 = nxt2[:]
    nxt0 = [0] * (S+1)
    nxt1 = [0] * (S+1)
    nxt2 = [0] * (S+1)
ans = dp2[S]
print(ans)
