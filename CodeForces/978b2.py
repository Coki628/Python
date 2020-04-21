# -*- coding: utf-8 -*-

"""
・ランレングス圧縮
・誰かのツイートで1行で書けるやつ前に見たことあったけど、やっと作った。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def RLE(data):
    from itertools import groupby

    return [(x, len(list(grp))) for x, grp in groupby(data)]    

N = INT()
S = input()

lre = RLE(S)
ans = 0
for s, cnt in lre:
    if s == 'x':
        ans += max(cnt-2, 0)
print(ans)
