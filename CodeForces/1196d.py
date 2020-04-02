# -*- coding: utf-8 -*-

"""
・制約弱い方。落ち着いてこれ先に書いて通した。
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

def check(S, T):
    res = 0
    for i in range(K):
        if S[i] != T[i]:
            res += 1
    return res

for _ in range(INT()):
    N, K = MAP()
    S = input()

    RGB = 'RGB'
    T1 = []
    for i in range(K):
        T1.append(RGB[i%3])
    T2 = []
    for i in range(K):
        T2.append(RGB[(i+1)%3])
    T3 = []
    for i in range(K):
        T3.append(RGB[(i+2)%3])

    ans = INF
    for i in range(N-K+1):
        j = i + K
        S2 = S[i:j]
        ans = min(ans, check(S2, T1))
        ans = min(ans, check(S2, T2))
        ans = min(ans, check(S2, T3))
    print(ans)
