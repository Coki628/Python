# -*- coding: utf-8 -*-

"""
・ランレングス圧縮
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
S = input()

for i in range(N):
    nxt = []
    cnt = 1
    for i, s in enumerate(S[1:], 1):
        if s == S[i-1]:
            cnt += 1
        else:
            nxt.append(str(cnt))
            nxt.append(S[i-1])
            cnt = 1
    nxt.append(str(cnt))
    nxt.append(S[-1])
    S = ''.join(nxt)
print(S)
