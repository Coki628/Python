# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！
・bitDP
・2^8通りの状態から2^8通りの状態に遷移させるため、そのままだとループの計算量合計は
　H*2^W*2^W*W=約5200万になってきついけど、実はあみだくじとして妥当じゃない集合は全て無視していいから、
　かなりの量を枝刈りできる。pythonでもAC1.4秒、pypyなら0.4秒。
・また想定解とは違うけど、これも典型に落とし込めた、と言うことで。
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

H, W, K = MAP()

# あみだくじとして妥当かの判定
def check(S):
    for i in range(W-1):
        # 1つでも隣り合う場所に横棒があればNG
        if S & 1<<i and S & 1<<(i+1):
            return False
    return True

# dp[i][S][j] := i段目まで見て、今の段の横棒の状態がSで、j本目の縦棒に行く通り数
dp = list3d(H+1, 1<<W, W+1, 0)
dp[0][0][1] = 1
for i in range(H):
    for S1 in range(1<<W):
        # 左端はただの番兵なので、立っていたらスキップ
        if S1 & 1:
            continue
        # あみだくじとして妥当でない集合は全てスキップ
        if not check(S1):
            continue
        for S2 in range(1<<W):
            if S2 & 1:
                continue
            if not check(S2):
                continue
            for j in range(1, W+1):
                # 左に棒があれば左に遷移
                if S2 & 1<<(j-1):
                    dp[i+1][S2][j-1] += dp[i][S1][j]
                    dp[i+1][S2][j-1] %= MOD
                # 右に棒があれば右に遷移
                elif S2 & 1<<j:
                    dp[i+1][S2][j+1] += dp[i][S1][j]
                    dp[i+1][S2][j+1] %= MOD
                # 棒がなければ真下に遷移
                else:
                    dp[i+1][S2][j] += dp[i][S1][j]
                    dp[i+1][S2][j] %= MOD
ans = 0
for S in range(1<<W):
    ans += dp[H][S][K]
    ans %= MOD
print(ans)
