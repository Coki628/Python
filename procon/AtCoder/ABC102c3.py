# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！
・三分探索
・絶対これは想定解じゃないだろうと思いつつも、他の方法が出なかったのでこれで通した。
・小数ではやったことあったけど、整数の三分探索は初めてだったので、
　これはこれでやってみて良かった。
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

N = INT()
A = LIST()

# bをmとした時の悲しさ値
def check(m):
    sm = 0
    for i, a in enumerate(A):
        sm += abs(a - (m+i+1))
    return sm

# 三分探索
low = -10**9
hi = 10**9
while low+2 < hi:
    m1 = (low*2+hi) // 3
    m2 = (low+hi*2) // 3
    res1 = check(m1)
    res2 = check(m2)
    # 今回は下に凸な関数なので値の小さい方に向かって狭めていく
    if res1 <= res2:
        hi = m2
    else:
        low = m1

print(min(check(m1), check(m2)))
