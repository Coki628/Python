# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・正しい方針に行き着いて良かった。
・結局、該当するのは2パターンしかない。
・AtCoderはこういう考察必須、方針決まれば実装軽いみたいなやつ多いよなぁ。
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

S = input()
N = len(S)

# 1つ先に同じ文字
for i in range(N-1):
    if S[i] == S[i+1]:
        print(i+1, i+2)
        exit()

# 2つ先に同じ文字
for i in range(N-2):
    if S[i] == S[i+2]:
        print(i+1, i+3)
        exit()

# どちらも該当なし
print(-1, -1)
