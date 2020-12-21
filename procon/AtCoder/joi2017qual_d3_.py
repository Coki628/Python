# -*- coding: utf-8 -*-

"""
参考：http://microbe.hatenablog.com/entry/2017/12/11/235414
　　　https://www.ioi-jp.org/joi/2017/2018-yo/2018-yo-t4-review.html
・公式解説分からなくて、保留にしてたやつ。
　たまたま別件で見つけた人様のブログ記事でやっと理解。
・ちょっと変則的な2次元DP
・求めるのは最大値だけど、最小ピースの最大値なのもあって、途中でminもmaxも使う。
・なので-INF初期化とかするとmin取る時バグるから、-1初期化にして、不正な遷移元はスキップ。
・各切れ目について最大と最小を取りながら、適切に遷移させていく。
・ループだけ見ると計算量N*sum(A)*N=50*50万*50=12億5000万で死ねるけど、
　実際はdp[i][j]==-1のとこ内側回さないから全然速い。
・pythonAC0.6秒、pypyAC0.3秒。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST(N)
MAX = sum(A)

acc = [0] + list(accumulate(A))
# dp[i][j] := i本の切れ目までの切り方を確定し，長さ最大のピースがjである時のこれまでの最小のピースの最大値
dp = list2d(N+1, MAX+1, -1)
dp[0][0] = INF
for i in range(N):
    for j in range(MAX):
        if dp[i][j] != -1:
            # 確定済のiより先で、全ての位置で切れ目を入れてみる
            for k in range(i+1, N+1):
                # min(今までの最小ピース, 今回入れる切れ目)
                mn = min(dp[i][j], acc[k]-acc[i])
                # max(今までの最大ピース, 今回入れる切れ目)
                mx = max(j, acc[k]-acc[i])
                # [今回の切れ目位置][今回含めての最大ピース] に 今回含めての最小ピース を遷移させる
                dp[k][mx] = max(dp[k][mx], mn)

ans = INF
# j == MAX は1回も切ってなくて条件を満たさないので、見るのはMAX-1まで
for j in range(MAX):
    if dp[N][j] != -1:
        ans = min(ans, j-dp[N][j])
print(ans)
