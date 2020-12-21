# -*- coding: utf-8 -*-

"""
参考：https://nanikaka.hatenadiary.org/entry/20120412/1334175198
　　　https://www.ioi-jp.org/joi/2010/2011-yo-prob_and_sol/2011-yo-t6/review/2011-yo-t6-review.html
・蟻本演習3-4-3
・bitDP、スライドbitDP
・直前Mマスの全状態をビットに持たせる。
・良い旗より悪い旗の方が数えやすいので、余事象で答えを求める。
・計算量20^2*2^20=約4億はつらすぎる。当然pypyでもTLE。
・C++でAC2.2秒(制約10秒)。やっぱすげぇ。
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
MOD = 10 ** 5

def deepcopy(li): return [x[:] for x in li]

N, M = MAP()
grid = []
qcnt = 0
for i in range(N):
    grid.append(list(input()))
    qcnt += grid[-1].count('?')

# dp[i][S] := 左隣の文字がJかどうかが(i=0or1)で、直前Mマスの状態がS(JOとなっているJ位置のビットを立てる)である通り数
dp = list2d(2, 1<<M, 0)
dp[0][0] = 1
# Mビット全埋め(1ビットずらしの遷移に使う)
msk = (1<<M) - 1
for i in range(N):
    for j in range(M):
        # 直前Mマスの状態を全通り確認する
        nxt = list2d(2, 1<<M, 0)
        for S in range(1<<M):
            if dp[0][S] != 0 or dp[1][S] != 0:
                if grid[i][j] == '?':
                    # Jを置く(左隣JフラグTrueに遷移)
                    nxt[1][(S<<1)&msk] += dp[0][S]
                    nxt[1][(S<<1)&msk] += dp[1][S]
                    # Oを置く(左隣JだったらJOが成立するのでビットを立てる)
                    nxt[0][(S<<1)&msk] += dp[0][S]
                    nxt[0][((S|1)<<1)&msk] += dp[1][S]
                    # 真上がJOのJ位置でなければ(右端は無条件OK)
                    if not S & (1<<(M-1)) or j == M-1:
                        # Iを置く
                        nxt[0][(S<<1)&msk] += dp[0][S]
                        nxt[0][(S<<1)&msk] += dp[1][S]
                elif grid[i][j] == 'J':
                    nxt[1][(S<<1)&msk] += dp[0][S]
                    nxt[1][(S<<1)&msk] += dp[1][S]
                elif grid[i][j] == 'O':
                    nxt[0][(S<<1)&msk] += dp[0][S]
                    nxt[0][((S|1)<<1)&msk] += dp[1][S]
                else:
                    if not S & (1<<(M-1)) or j == M-1:
                        nxt[0][(S<<1)&msk] += dp[0][S]
                        nxt[0][(S<<1)&msk] += dp[1][S]
                nxt[0][(S<<1)&msk] %= MOD
                nxt[1][(S<<1)&msk] %= MOD
                nxt[0][((S|1)<<1)&msk] %= MOD
        dp = deepcopy(nxt)

# 良い旗の総数 = 全通り - 悪い旗の総数
sm = sum(dp[0]) + sum(dp[1])
sm %= MOD
print((pow(3, qcnt, MOD) - sm)%MOD)
