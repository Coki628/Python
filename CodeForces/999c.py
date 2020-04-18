# -*- coding: utf-8 -*-

"""
・文字毎にどこで出てくるかindex管理する。
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

N, K = MAP()
S = [ord(s)-97 for s in input()]

idxs = [[] for i in range(26)]
for i in range(N):
    idxs[S[i]].append(i)
for c in range(26):
    idxs[c] = idxs[c][::-1]

used = [False] * N
for _ in range(K):
    for c in range(26):
        if idxs[c]:
            i = idxs[c].pop()
            used[i] = True
            break

ans = []
for i in range(N):
    if not used[i]:
        ans.append(chr(S[i]+97))
print(''.join(ans))
