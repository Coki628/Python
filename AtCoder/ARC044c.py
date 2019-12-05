# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc044
　　　https://atcoder.jp/contests/arc044/submissions/8380174
・自力ならず。。良い方針全然出なかった。
・縦横独立に考える。これは本当に何度も出てるポイントなのに、捉えられなかった。
・10万*10万グリッドとか絶対それを疑うべきなのに。なんか複合的に絡む気がしてしまった。。
・DPは配列としては座標位置を持つんだけど、更新のループは時間軸のみで、位置では回さない。
　なぜなら更新するのはビームが来てる座標だけだから。
　他はdp[i+1][j] = dp[i][j]的なことが起きてるんだけど、見る必要はない。
・自力実装でAC後、ちょっとだけ改修。番兵をうまく入れて条件分岐をなくした。
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

T = 10 ** 5
W, H, Q = MAP()

# 時間毎に縦ビームと横ビームをまとめておく
TW = [[] for i in range(T+1)]
TH = [[] for i in range(T+1)]
for i in range(Q):
    t, d, x = MAP()
    if d == 0:
        TW[t].append(x)
    else:
        TH[t].append(x)

def calc(N, lists):
    # dp[i] := 座標iにいる時の最小移動距離
    dp = [0] * (N+2)
    # 移動不可の位置に番兵
    dp[0] = dp[N+1] = INF
    for li in lists:
        li.sort()
        # 左に避ける
        for x in li:
            dp[x] = dp[x-1] + 1
        # 右に避ける
        for x in li[::-1]:
            dp[x] = min(dp[x], dp[x+1] + 1)
    return min(dp)

ans = 0
# 縦横は完全に独立に考えられる
ans += calc(W, TW)
ans += calc(H, TH)
if ans < INF:
    print(ans)
else:
    print(-1)
