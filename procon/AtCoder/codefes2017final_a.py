# -*- coding: utf-8 -*-

"""
・300点自力AC
・正規表現
"""

import sys, re

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

# 先頭、末尾、0回または1回
if re.match(r'^(A?)KIH(A?)B(A?)R(A?)$', S):
    YES()
else:
    NO()
