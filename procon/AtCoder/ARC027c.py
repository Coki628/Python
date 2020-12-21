# -*- coding: utf-8 -*-

"""
・自力TLE。多分、方針は合ってる。
・最初MLE → 配列をi方向2つだけにしてメモリ削減
・それでまだTLE → ちょっと考えて、枝刈りを入れた。
・それでもまだTLE → C++でAC0.1秒。
・計算量600*300*600=1億800万、、まあ、これはしょうがないか。。
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

X, Y = MAP()
N = INT()
TH = []
for i in range(N):
    t, h = MAP()
    TH.append((t, h))

M = X + Y
# dp[i][j][k] := i個目のトッピングまで見て、スペシャルチケットj枚、通常チケットk枚を使った時の、嬉しさの最大値
dp = list3d(2, X+1, M+1, 0)
for i in range(N):
    cur = i % 2
    nxt = (i+1) % 2
    t, h = TH[i]
    for j in range(X+1):
        for k in range(M+1):
            # 枝刈り
            if j+k > M:
                break
            dp[nxt][j][k] = max(dp[nxt][j][k], dp[cur][j][k])
            if j+1 <= X and k+t-1 <= M:
                # スペシャルチケットは常に1枚だけ使うとする
                dp[nxt][j+1][k+t-1] = max(dp[nxt][j+1][k+t-1], dp[cur][j][k]+h)

ans = 0
for j in range(X+1):
    for k in range(M+1):
        if j+k > M:
            break
        # 両チケットの和がM以内なら、通常チケットで数えた分をスペシャルチケットで代替できるので答えとして使える
        ans = max(ans, dp[N%2][j][k])
print(ans)
