# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・また典型っぽい感じのナップザック系DP
・饅頭の詰め方に制限はないので、大きい方から貪欲。
・箱についてはDPする。両方合わせて最適になる所を見つければいい。
・O(NM)が500万だから制約1秒どうかなーと思ったけど案外余裕のpypyAC0.4秒。
　外ループが500で小さいからかなー？まあいいんだけど。
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

M, N = MAP()
P = LIST(M)
CE = []
for _ in range(N):
    c, e = MAP()
    CE.append((c, e))

# dp[i][j] := i個目の箱まで見て、饅頭をj個入れたい時の箱購入の最小コスト
dp = list2d(N+1, M+1, INF)
dp[0][0] = 0
for i, (c, e) in enumerate(CE):
    for j in range(M+1):
        # この箱を使わない
        dp[i+1][j] = min(dp[i+1][j], dp[i][j])
        # この箱を使う
        dp[i+1][min(j+c, M)] = min(dp[i+1][min(j+c, M)], dp[i][j] + e)

# 入れる饅頭は自由なので大きい方から
P.sort(reverse=1)
P = [0] + P
val = 0
ans = -INF
for j in range(M+1):
    val += P[j]
    # 饅頭をj個入れた利益 - そのための箱購入コスト
    ans = max(ans, val - dp[N][j])
print(ans)
