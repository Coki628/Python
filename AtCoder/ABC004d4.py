# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc004
　　　http://hatanas.hatenablog.com/entry/abc004
・前からminmaxDP
・状況が整理できたら、すごく素直なナップザップDP的遷移だった。
・計算量はメモ化再帰と同じで座標位置の数*マーブルの総数=1000*900=90万
　python0.9秒AC。前からDPのが速いと思ったのに遅くなった。
　多分、向こうは条件分岐で余計な遷移を減らせてるんだと思う。
・!=INFの条件分岐はなくても問題なく通るけど、なくしたら1.4秒まで遅くなった。
　calcの呼び出しとかで結構かかってるんだろうか。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

R, G, B = MAP()
N = R + G + B

# どのマーブルを使うか
def calc(cur, cnt):
    # R
    if N - cnt > G + B:
        return abs(400-cur)
    # G
    elif N - cnt > B:
        return abs(500-cur)
    # B
    else:
        return abs(600-cur)

# dp[i][j] := マーブル使用回数jの時の、位置iまでの最小操作回数
dp = list2d(1001, N+1, INF)
# 初期化：使用回数0なら移動回数も0
for i in range(1001):
    dp[i][0] = 0

for i in range(1000):
    for j in range(N+1):
        if dp[i][j] != INF:
            # マーブルを置かない場合の遷移
            dp[i+1][j] = min(dp[i+1][j], dp[i][j])
            if j < N:
                # マーブルを置く場合の遷移
                dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + calc(i, j))
mn = INF
for i in range(1001):
    # N個置き終わった中での最小値を取る
    mn = min(mn, dp[i][N])
print(mn)
