# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・XOR系、XORの特性をうまく使ってやる。
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

A, B = MAP()

# 最初が奇数なら面倒なので分けておく
if A % 2 == 1:
    a = A
    A += 1
else:
    a = 0

# 偶奇偶奇..の連続した区間について、2つで1、4つで0になるので、それをうまく使って場合分け
seg = B - A + 1
if seg % 4 == 0:
    b = 0
elif seg % 4 == 1:
    b = B
elif seg % 4 == 2:
    b = 1
else:
    b = 1 ^ B

print(a ^ b)
