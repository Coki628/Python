# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc161/editorial.pdf
・自力ならず。。
・前後から貪欲
・解法聞いたらまあ確かに、って感じ。実装はすぐ。
"""

import sys
from itertools import accumulate

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

N, K, C = MAP()
S = input()

L = [0] * K
cur = 0
idx = 0
while idx < K:
    if S[cur] == 'o':
        L[idx] = cur
        idx += 1
        cur += C + 1
    else:
        cur += 1
R = [0] * K
cur = N - 1
idx = K - 1
while idx >= 0:
    if S[cur] == 'o':
        R[idx] = cur
        idx -= 1
        cur -= C + 1
    else:
        cur -= 1

ans = []
for i in range(K):
    if L[i] == R[i]:
        ans.append(L[i])
[print(a+1) for a in ans]
