# -*- coding: utf-8 -*-

"""
・スタックで頑張った。WA。。
"""

import sys
from collections import deque
from itertools import accumulate

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

N = INT()
A = LIST()
M = max(A)

stack = deque()

gaps = []
s = INF
for a in A:
    if not len(stack) or stack[-1] > a:
        stack.append(a)
    elif stack[-1] < a:
        nxt = a
        while len(stack) and stack[-1] <= nxt:
            nxt = stack.pop()
        if len(stack):
            s = min(s, nxt)
            t = min(stack[-1], a)
            gaps.append((s, t))
            s = t

imos = [0] * (M+2)
for s, t in gaps:
    imos[s] += 1
    imos[t] -= 1
imos = list(accumulate(imos))

print(max(imos)+1)
