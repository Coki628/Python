# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-thanks-festival-2017/editorial.pdf
　　　http://kurkur.hatenablog.com/entry/2018/02/08/022629
・蟻本演習3-2-7
・半分全列挙、bitDP、最大独立集合問題
・DPの準備パートが大変。3つ作る。
・辺の有無を反転させると最大クリーク問題になるとのこと。
・N=40だから、計算量2^20*20=2000万ちょっとを3回回すのできついかもと思ったけど、pypyAC1.0秒。
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
INF = float('inf')
MOD = 10 ** 9 + 7

def bit_count(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

N, M = MAP()
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    a -= 1
    b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

N1 = N // 2
N2 = N - N1
# dp1[S] := 頂点集合Sが独立集合かどうか(前半分)
dp1 = [1] * (1<<N1)
# 繋がっている2頂点集合についてFalseにする
for u in range(N1):
    for v in nodes[u]:
        if v < N1:
            dp1[1<<u|1<<v] = 0
# Falseにした2頂点を含む集合をFalseにしていく
for i in range(1<<N1):
    if not dp1[i]:
        for j in range(N1):
            dp1[i|1<<j] = 0

# dp2[S] := 前半分の部分集合Sについて、Sと辺で繋がっていない後半分の頂点集合
dp2 = [0] * (1<<N1)
# 前半の頂点なしなら、後半の全ての頂点は繋がっていない(全ビット1)
dp2[0] = (1<<N2) - 1
# 1頂点集合の初期化
for u in range(N1):
    dp2[1<<u] = (1<<N2) - 1
    for v in nodes[u]:
        if v >= N1:
            # 前半から後半に繋がる辺について、該当する後半頂点のビットを0にする
            dp2[1<<u] ^= 1<<(v-N1)
for i in range(1<<N1):
    for j in range(N1):
        # 両方の集合で繋がっていない(ビットが1の)状態の頂点だけを残して遷移する
        dp2[i|1<<j] = dp2[i] & dp2[1<<j]

# dp3[S] := 頂点集合Sの最大独立集合のサイズ(後半分)
dp3 = [1] * (1<<N2)
# まずはdp1と同じ手順で独立集合かどうかの判定を作る
for u in range(N1, N):
    for v in nodes[u]:
        if v >= N1:
            dp3[1<<(u-N1)|1<<(v-N1)] = 0
for i in range(1<<N2):
    if not dp3[i]:
        for j in range(N2):
            dp3[i|1<<j] = 0
# 独立集合にTrueの代わりにサイズ(頂点数)を入れる
for i in range(1<<N2):
    if dp3[i]:
        dp3[i] = bit_count(i)
# 最大にしたいので、部分集合からmaxを取っていく
for i in range(1<<N2):
    for j in range(N2):
        dp3[i|1<<j] = max(dp3[i|1<<j], dp3[i])

ans = 0
for s1 in range(1<<N1):
    # 集合s1が独立集合かどうか
    if dp1[s1]:
        # s1と繋がっていない後半の頂点集合U
        U = dp2[s1]
        # 集合Uの最大独立集合s2
        s2 = dp3[U]
        ans = max(ans, bit_count(s1) + s2)
print(ans)
