# -*- coding: utf-8 -*-

"""
・蟻本演習、自力AC！
・半分前列挙、ナップザックでやったの思い出しながらやった。
・今回はテーマとしてやったからいいけど、N=1000の形でこの方針に気付けるかどうかだな。。
"""

import sys
from bisect import bisect_right

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
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
A = LIST(N)

# 半分全列挙
A1 = [0] + A[:N//2]
A2 = [0] + A[N//2:]
S1 = set()
S2 = set()
for i in range(len(A1)):
    for j in range(i, len(A1)):
        S1.add(A1[i]+A1[j])
for i in range(len(A2)):
    for j in range(i, len(A2)):
        S2.add(A2[i]+A2[j])

S1 = sorted(S1)
S2 = sorted(S2)
ans = 0
# S1とS2、S1とS1、S2とS2のそれぞれで最適な組を探す
for a in S1:
    idx = bisect_right(S2, M-a) - 1
    if idx == -1:
        continue
    ans = max(ans, a+S2[idx])
for a in S1:
    idx = bisect_right(S1, M-a) - 1
    if idx == -1:
        continue
    ans = max(ans, a+S1[idx])
for a in S2:
    idx = bisect_right(S2, M-a) - 1
    if idx == -1:
        continue
    ans = max(ans, a+S2[idx])
print(ans)
