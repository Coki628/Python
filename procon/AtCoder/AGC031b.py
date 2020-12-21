# -*- coding: utf-8 -*-

"""
参考：https://betrue12.hateblo.jp/entry/2019/03/17/011847
　　　https://scrapbox.io/ganariya/AtCoderGrandContest031_B%E5%95%8F%E9%A1%8C700%E7%82%B9%E3%80%8CReversi%E3%80%8D
・アルメリアさんの説明が一番しっくり来た。
・考察が要る1次元DP、こういうのちゃんと自分で導けるようになりたいね。
・遷移は変色させない場合とさせる場合の2つだけ。
・変色させる遷移が可能な時は、現在位置を右端として、左端になる場所から遷移させる。
・任意の位置を右端としてどこが左端になるかは、前もって調べておける。
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

N = INT()
A = LIST(N)

# 同色が直前に出現した位置を取っておく
prev = [INF] * N
D = {}
for i, a in enumerate(A):
    if a in D:
        prev[i] = D[a]
    D[a] = i

# dp[i] := 添字iが右端となる区間までの通り数
dp = [0] * N
dp[0] = 1
for i in range(1, N):
    # 1つ前と変化しない遷移は必ずある
    dp[i] += dp[i-1]
    # 左隣りが違う色でかつ、自分より左に同色があるなら、挟んで変色させる遷移をする
    if A[i-1] != A[i] and prev[i] != INF:
        # 挟む左端となる場所から遷移させる
        dp[i] += dp[prev[i]]
    dp[i] %= MOD
print(dp[N-1])
