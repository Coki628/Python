# -*- coding: utf-8 -*-

"""
参考：http://arc057.contest.atcoder.jp/data/arc/057/editorial.pdf
　　　https://atcoder.jp/contests/arc057/submissions/6668304
・とりあえず公式解説だけ見ての自力実装
　→WAが取れずに人のやつと比較したりした結果、>が>=になってただけだった…。
・minmaxDP + 変化する量を計算する系
・式変形：今回必要な勝ち数をXとして、
　そこまでの総勝ち数/そこまでの総ゲーム数 < (そこまでの総勝ち数+X)/(そこまでの総ゲーム数+今回のゲーム数)
　をXについて解く。
・計算量は2000^2=400万、pythonTLEでpypy0.5秒AC。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def LIST2(n): return [INT() for i in range(n)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST2(N)

game_cnts = [0] + list(accumulate(A))

# 総ゲーム数 == 勝利数 は例外処理
if game_cnts[-1] == K:
    print(1)
    exit()

# 今回機嫌を良くする(勝率を上げる)ために必要な勝利数を計算
def calc(total_game, total_win, cur_game):
    # 最初は絶対1勝必要
    if total_game == 0: return 1
    a, b, c = total_game, total_win, cur_game
    # 式変形の結果こうなる
    ans = b*(a+c)/a-b
    # 0未満なら1回も勝たなくていい
    if ans < 0: return 0
    # 切り捨て+1でansより大きい最小の整数
    ans = int(ans) + 1
    # 今回のゲーム数より多かったら無理
    if ans > cur_game: return INF
    # 問題なければ答えを返却
    return ans

# dp[i][j] := i日目までにj回機嫌が良くなった時の最小の勝利数
dp = list2d(N+1, N+1, INF)
dp[0][0] = 0

for i in range(N):
    for j in range(N+1):
        # INFになったらその先は遷移させる意味ない
        if dp[i][j] == INF: break
        # 今回機嫌が良くならない遷移
        dp[i+1][j] = min(dp[i+1][j], dp[i][j])
        if j <= N:
            # 今回機嫌を良くする遷移
            dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j]+calc(game_cnts[i], dp[i][j], A[i]))

mx = 0
for j in range(N+1):
    # K以下の勝利数で得られる最大の機嫌が良くなった回数
    if dp[N][j] <= K:
        mx = max(mx, j)
print(mx)
