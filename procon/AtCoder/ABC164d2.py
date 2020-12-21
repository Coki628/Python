# -*- coding: utf-8 -*-

"""
参考：みんなのツイート
・別解。実はDP数え上げで通るって話。
・桁DP系
・状態3つに分けた。(本当はもっと省略できた。)
・各遷移では、遷移前の値(添字j)*10 + 今回の値a を繰り返せば、うまく値の桁が動いてよしなにいける。
・この方針で遷移作る時は、多分後ろからより前(大きい桁)からやった方が良さそう。
・計算量20万*2019で4億くらいはさすがにpypyでもきつかったので、C++でAC確認。
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

A = list(map(int, input()))
N = len(A)
mod = 2019

# dp0[i][j] := i桁目まで見て、modでjになる通り数(区間開始前)
dp0 = list2d(2, mod, 0)
# dp1[i][j] := i桁目まで見て、modでjになる通り数(区間継続中)
dp1 = list2d(2, mod, 0)
# dp2[i][j] := i桁目まで見て、modでjになる通り数(区間終了後)
dp2 = list2d(2, mod, 0)
dp0[0][0] = 1
for i, a in enumerate(A):
    # まだ区間を開始しない
    dp0[(i+1)&1][0] = dp0[i&1][0]
    # ここから区間を開始
    dp1[(i+1)&1][a] = dp0[i&1][0]
    for j in range(mod):
        nxt = (j*10+a) % mod
        # 区間を継続
        dp1[(i+1)&1][nxt] += dp1[i&1][j]
        # ここで区間を終了
        dp2[(i+1)&1][nxt] += dp1[i&1][j]
        # 既に区間終了済
        dp2[(i+1)&1][j] += dp2[i&1][j]
    dp0[i&1] = [0] * mod
    dp1[i&1] = [0] * mod
    dp2[i&1] = [0] * mod
# N桁全部見て、区間終了済で、mod 2019が0になる通り数
ans = dp2[N&1][0]
print(ans)
