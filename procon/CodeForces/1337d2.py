# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1337/submission/76879157
・終了後、頑張ってみたけど自力ならず。。
・二分探索、入出力
・1点固定で残り2つをにぶたん。ほぼ正解に近いものは作ったんだけどWA。
・ももはらさんの提出から、軸足にするのRGB全部試せばいい感じがしたので(Rからしかやってなかった)、やってみたら無事AC。
・また入出力がきついみたいで、bufferつけないとTLEした。数値はこれでいいけど、文字列の時困るよなー。
・計算量は10万のNにlogと定数くらいなんだけど、pypyAC2.0秒。(制約3秒)
・同じこと3回書くの冗長だなぁと思ってまとめた。
・定数3とはいえ配列もネストも深くなるし、ちょっと遅くなるかと思ったんだけど意外と変わらなかった。
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

    RGB = [[]] * 3
    RGB[0] = sorted(set(R))
    RGB[1] = sorted(set(G))
    RGB[2] = sorted(set(B))
    N = [0] * 3
    for i in range(3):
        N[i] = len(RGB[i])
    ans = INF
    for c in range(3):
        for i in range(N[c%3]):
            x = RGB[c%3][i]
            idxy = bisect_left(RGB[(c+1)%3], x)
            idxz = bisect_left(RGB[(c+2)%3], x)
            for j in range(max(idxy-1, 0), min(idxy+1, N[(c+1)%3])):
                for k in range(max(idxz-1, 0), min(idxz+1, N[(c+2)%3])):
                    y = RGB[(c+1)%3][j]
                    z = RGB[(c+2)%3][k]
                    ans = min(ans, (x-y)**2 + (y-z)**2 + (z-x)**2)
    print(ans)
