# -*- coding: utf-8 -*-

"""
・さくっと自力AC、マッピング書くの疲れた。
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
words = input().split()

d = {
    'b': 1, 'c': 1,
    'd': 2, 'w': 2,
    't': 3, 'j': 3,
    'f': 4, 'q': 4,
    'l': 5, 'v': 5,
    's': 6, 'x': 6,
    'p': 7, 'm': 7,
    'h': 8, 'k': 8,
    'n': 9, 'g': 9,
    'z': 0, 'r': 0,
}

ans = []
for word in words:
    word = word.lower()
    num = ''
    for c in word:
        if c in d:
            num += str(d[c])
    if num != '':
        ans.append(num)
print(*ans)
