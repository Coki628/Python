# -*- coding: utf-8 -*-

"""
参考：ツイート
・自力ならず。。
・回文系
・回文でかつK文字毎に同じ文字列にする、なんだけど、
　結果的にこれはK文字毎に同じ回文にすれば、全体も回文になってうまくいく。
　このことに気付けなかった。。ていうか、途中ちょっと考えた気がするんだけど、
　なんかKが奇数の時に真ん中ダメじゃんとか勝手に思い込んでしまってこの線を消してた。。
　全然ダメじゃないね。。回文真ん中で切れても左右対称なんだからOKね。。
・それに気付ければ、後は各K内の回文で対になる部分毎に文字の種類を集計して、
　一番多いやつに合わせる、でOK。
・これは気付きたかったな。。みんな通す訳だ。。
"""

import sys
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K = MAP()
    S = input()

    C = [Counter() for i in range(ceil(K, 2))]
    for i in range(0, N, K):
        for j in range(K//2):
            a, b = S[i+j], S[i+K-j-1]
            C[j][a] += 1
            C[j][b] += 1
        if K % 2 == 1:
            a = S[i+K//2]
            C[K//2][a] += 1
    
    ans = 0
    for c in C:
        ans += sum(c.values()) - max(c.values())
    print(ans)
