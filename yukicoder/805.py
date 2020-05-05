# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・2重ループいけるので、外ループで真ん中決め打ちで内ループで左右探しに行く。
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

ans = 0
for j in range(1, N-1):
    i = j - 1
    k = j + 1
    while i >= 0 and k < N:
        if S[i] == 'U' and S[j] == 'M' and S[k] == 'G':
            ans += 1
        i -= 1
        k += 1
print(ans)
