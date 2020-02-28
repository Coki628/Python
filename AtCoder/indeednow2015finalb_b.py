# -*- coding: utf-8 -*-

"""
・累積和、区間管理
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

N = INT()

acc = [0] * (N*2+1)
A = [0] * N
for i in range(N):
    s, t = MAP()
    A[i] = (s, t)
    acc[s] += 1
# 出社時間で累積和
acc = list(accumulate(acc))

ans = []
for s, t in A:
    # 自分のいる間に、何人の出社があったか(自分を除く)
    ans.append(acc[t] - acc[s-1] - 1)
[print(a) for a in ans]
