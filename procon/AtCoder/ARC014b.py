# -*- coding: utf-8 -*-

"""
・さくっと自力AC
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
words = []
for i in range(N):
    words.append(input())

used = set()
used.add(words[0])
nxt = words[0][-1]
for i in range(1, N):
    word = words[i]
    if word in used or word[0] != nxt:
        if i%2 == 1:
            print('WIN')
        else:
            print('LOSE')
        exit()
    nxt = word[-1]
    used.add(word)
print('DRAW')
