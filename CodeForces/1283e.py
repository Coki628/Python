# -*- coding: utf-8 -*-

"""
・自力ならず。。
・貪欲を詰める。
・最小の方はだいたい合ってたけど、最大の方がうまくいかなかった。
・最終的に、左から貪欲に、左と自分位置が0だったら1ずつ残す、残ったら右に渡す、で通った。
"""

import sys
from collections import Counter

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

N = INT()
A = LIST()

C = Counter(A)

def check1(C):
    C2 = [0] * (N+2)
    i = 0
    while i < N+1:
        if C[i] > 0:
            C2[i+1] = C[i] + C[i+1] + C[i+2]
            C[i] = C[i+1] = C[i+2] = 0
            i += 3
        else:
            C2[i] = C[i]
            C[i] = 0
            i += 1
    return C2

def check2(C):
    C2 = [0] * (N+2)
    i = 0
    while i < N:
        if C[i+1] > 0 and C2[i] == 0:
            C2[i] += 1
            C[i+1] -= 1
        if C[i+1] > 0 and C2[i+1] == 0:
            C2[i+1] += 1
            C[i+1] -= 1
        C2[i+2] += C[i+1]
        C[i+1] = 0
        i += 1
    return C2

res1 = check1(C.copy())
res2 = check2(C.copy())
ans1 = N + 2 - res1.count(0)
ans2 = N + 2 - res2.count(0)
print(ans1, ans2)
