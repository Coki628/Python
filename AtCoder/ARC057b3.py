# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/arc057/submissions/6091696
・minmaxDP + 変化する量を計算する系
・maspyさんのやつ見てnumpy実装作る。
・以下、得られた知見を列挙
　・dtype=np.int64で整数型初期化
　・.cumsum()で累積和一撃
　・A[1:] = B とかやればAの二番目以降にまとめてBを突っ込んだりできる
　・np.fullで任意の値で初期化
　・np.minimumで、配列まとめて最小値取得 ←これやばい
　・np.searchsortedは二分探索、bisectみたいなもん
・肝はやっぱりDPのループで配列まとめて遷移するところで、
　最後の二分探索は普通に全探索でやっても速度変わらず0.3秒AC。
"""

import sys
import numpy as np

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
A = np.array(LIST2(N), dtype=np.int64)

game_cnts = np.zeros(N+1, dtype=np.int64)
game_cnts[1:] = A.cumsum()

# 総ゲーム数 == 勝利数 は例外処理
if game_cnts[-1] == K:
    print(1)
    exit()

# dp[i][j] := i日目までにj回機嫌が良くなった時の最小の勝利数
INF = 2**31-1
dp = np.full((N+1, N+1), INF, dtype=np.int64)
dp[0,0] = 0
dp[1,0] = 0
# 最初は絶対1勝必要
dp[1,1] = 1

for i in range(1, N):
    # 今回機嫌が良くならない遷移
    dp[i+1] = dp[i]
    # 今回機嫌を良くする遷移(b*(a+c)/a)
    a, c = game_cnts[i], A[i]
    dp[i+1,1:] = np.minimum(dp[i+1,1:], dp[i,:-1]*(a+c)//a+1)

# K以下の勝利数で得られる最大の機嫌が良くなった回数
ans = np.searchsorted(dp[N], K, side='right') - 1
print(ans)
