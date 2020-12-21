# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1337/submission/76879157
・終了後、頑張ってみたけど自力ならず。。
・二分探索、入出力
・1点固定で残り2つをにぶたん。ほぼ正解に近いものは作ったんだけどWA。
・ももはらさんの提出から、軸足にするのRGB全部試せばいい感じがしたので(Rからしかやってなかった)、やってみたら無事AC。
・また入出力がきついみたいで、bufferつけないとTLEした。数値はこれでいいけど、文字列の時困るよなー。
・計算量は10万のNにlogと定数くらいなんだけど、pypyAC2.0秒。(制約3秒)
"""

import sys
from bisect import bisect_left

def input(): return sys.stdin.buffer.readline().strip()
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N1, N2, N3 = MAP()
    R = LIST()
    G = LIST()
    B = LIST()

    R = sorted(set(R))
    G = sorted(set(G))
    B = sorted(set(B))
    N1, N2, N3 = len(R), len(G), len(B)
    ans = INF
    for i in range(N1):
        r = R[i]
        idxg = bisect_left(G, r)
        idxb = bisect_left(B, r)
        for j in range(max(idxg-1, 0), min(idxg+1, N2)):
            for k in range(max(idxb-1, 0), min(idxb+1, N3)):
                g = G[j]
                b = B[k]
                ans = min(ans, (r-g)**2 + (g-b)**2 + (b-r)**2)
    for i in range(N2):
        g = G[i]
        idxr = bisect_left(R, g)
        idxb = bisect_left(B, g)
        for j in range(max(idxr-1, 0), min(idxr+1, N1)):
            for k in range(max(idxb-1, 0), min(idxb+1, N3)):
                r = R[j]
                b = B[k]
                ans = min(ans, (r-g)**2 + (g-b)**2 + (b-r)**2)
    for i in range(N3):
        b = B[i]
        idxg = bisect_left(G, b)
        idxr = bisect_left(R, b)
        for j in range(max(idxg-1, 0), min(idxg+1, N2)):
            for k in range(max(idxr-1, 0), min(idxr+1, N1)):
                g = G[j]
                r = R[k]
                ans = min(ans, (r-g)**2 + (g-b)**2 + (b-r)**2)
    print(ans)
