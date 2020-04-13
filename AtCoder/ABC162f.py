# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc162/editorial.pdf
・自力ならず。。
・状態ちょこっとだけ持つDP(いわゆる耳DP？)
・使った要素数を全部持つとN^2で死ぬけど、実は必要なのは全体との差分で見れば4つくらい。
・これと直前を使ったかどうかを合わせて持っておけば、O(N)でDPできる。
・計算量は20万のN、なんかdefaultdict使ったら遅かった。pythonTLE,pypyAC0.6秒。
"""

import sys
from collections import defaultdict

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

N = INT()
A = LIST()

# dp0[i][j] := i個目の要素まで見て、選んだ要素-選ばなかった要素=jで、直前を使っていない時の最大値
dp0 = [defaultdict(lambda: -INF) for i in range(N+1)]
# dp1[i][j] := i個目の要素まで見て、選んだ要素-選ばなかった要素=jで、直前を使った時の最大値
dp1 = [defaultdict(lambda: -INF) for i in range(N+1)]
dp0[0][0] = 0
for i, a in enumerate(A):
    for j in range(-2, 2):
        if j != -2:
            # 今回のaを使わない
            dp0[i+1][j-1] = max(dp0[i+1][j-1], dp0[i][j])
            dp0[i+1][j-1] = max(dp0[i+1][j-1], dp1[i][j])
        if j != 1:
            # 今回のaを使う
            dp1[i+1][j+1] = max(dp1[i+1][j+1], dp0[i][j] + a)
# 最終的に欲しい位置は偶奇で場合分け
if N % 2 == 0:
    # 偶数の時は使った数と使わなかった数が同じ
    ans = max(dp0[N][0], dp1[N][0])
else:
    # 奇数の時は使わなかった数が1つ多い
    ans = max(dp0[N][-1], dp1[N][-1])
print(ans)
