# -*- coding: utf-8 -*-

"""
・400点自力AC！
・回文系
・回文にならないようにする。
"""

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

S=input()

c=Counter(S)

# abcabcabcab..みたく続けたいので、どれかに2つ以上の差があると良くない
if abs(c['a']-c['b'])<=1 and abs(c['b']-c['c'])<=1 and abs(c['c']-c['a'])<=1:
    YES()
else:
    NO()
