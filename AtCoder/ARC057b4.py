# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-4、自力AC！
・目的の答えを添字にして、大きい値を中身に詰めて、最後に最適な添字を探す系(長い…)
・確か初見の時、全然解けなかったやつだから、今回過去のやつ見ないでちゃんと解けて嬉しい。
　まあ、時間はかかったけど。。
"""

import sys
from itertools import accumulate

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

N, K = MAP()
A = LIST(N)
total = [0] + list(accumulate(A))

# 例外処理
if sum(A) == K:
    print(1)
    exit()

# 必要な勝利数を算出
def calc(cur_total, prev_total, prev_win):
    # 初回は1勝必要
    if prev_total == 0:
        return 1
    if prev_win == INF:
        return INF
    # cur_win > prev_win * cur_total / prev_total に対応
    cur_win = prev_win * cur_total // prev_total + 1
    # 今回の試合数より多くは勝てない
    if cur_win - prev_win > cur_total - prev_total:
        return INF
    return cur_win - prev_win

# dp[i][j] := i日間中j日機嫌が良かった時の、必要な勝利数の最小値
dp = list2d(N+1, N+1, INF)
dp[0][0] = 0
for i in range(N):
    for j in range(i+1):
        # i+1日目に機嫌が良くならない遷移(1回も追加で勝つ必要はない)
        dp[i+1][j] = min(dp[i+1][j], dp[i][j])
        # 機嫌を良くする遷移(必要勝利数を追加する)
        dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + calc(total[i+1], total[i], dp[i][j]))

ans = 0
for i in range(N+1):
    if dp[N][i] <= K:
        ans = max(ans, i)
print(ans)
