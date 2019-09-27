# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2016/2017-yo/2017-yo-t4/review/2017-yo-t4-review.html
・累積和、bitDP
・順列のN!はbitDPで2^Nにできる。それはちょっと考えたけど、うまく形をイメージできなかった。
　でも解説読んだ後は、自分の別のbitDPコード(ABC041d2)ちょっと参考にするくらいで自力実装できたからまだ良かったかな。
・計算量は累積和のN*MとbitDPの2^M*Mで200万+2000万の2200万、pythonTLE(制限10秒なのに)でpypyAC1.7秒。
"""

import sys
from itertools import accumulate
from collections import Counter

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

N, M = MAP()
A = [a-1 for a in LIST(N)]

# 現在位置の区間和をO(1)取得するために、各Mについて累積和しておく
acc = list2d(M, N+1, 0)
for i, a in enumerate(A):
    # 区間にある個数が欲しいので、あるorないで2値化
    acc[a][i+1] = 1
for i in range(M):
    acc[i] = list(accumulate(acc[i]))

C = Counter(A)
# dp[i] := 集合iの時に取り出す必要があるぬいぐるみの最小値
dp = [INF] * (1<<M)
# 初期化(一番左に並べるぬいぐるみ)
for i in range(M):
    s, t = 0, C[i]
    diff = C[i] - (acc[i][t]-acc[i][s])
    dp[1<<i] = diff

for i in range(1<<M):

    # 現在の集合iの状態での開始indexを特定する
    s = 0
    for j in range(M):
        # jが含まれていたら、その分開始位置に足す
        if (i>>j) & 1:
            s += C[j]

    # 集合iにぬいぐるみjを追加する
    for j in range(M):
        if (i>>j) & 1:
            continue
        t = s + C[j]
        # 移動が必要な個数 = aの総数 - 区間に含まれる個数(動かさなくていい個数)
        diff = C[j] - (acc[j][t]-acc[j][s])
        dp[i+(1<<j)] = min(dp[i+(1<<j)], dp[i]+diff)

print(dp[(1<<M)-1])
