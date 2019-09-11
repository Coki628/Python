# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・階乗の前計算しない版
・最大1万の階乗があるけど、その場合1回しか出ないから前計算いらないかなーって思ったけど、
　やっぱり要らなかった。こっちでも速度変わらず0.03秒。
"""

import sys
from itertools import accumulate
from collections import Counter
from math import factorial

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
A = [INT() for i in range(N)]

A.sort()
sm = sum(accumulate(A))

C = Counter(A)
cnt = 1
for v in C.values():
    cnt *= factorial(v)
    cnt %= MOD

print(sm)
print(cnt)
