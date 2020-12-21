# -*- coding: utf-8 -*-

"""
・自力AC
・XOR系
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

# とりあえず立っているビットを集計
C = Counter()
for a in A:
    i = 0
    while (a >> i) > 0:
        if a & 1<<i:
            C[i] += 1
        i += 1

mx = 0
idx = -1
for i, a in enumerate(A):
    sm = 0
    j = 0
    while (a >> j) > 0:
        # このビットが立っているのが自分1つだけなら残せる
        if a & 1<<j and C[j] == 1:
            sm += 1<<j
        j += 1
    # 残せる合計の最も多い要素のindexを覚えておく
    if sm > mx:
        mx = sm
        idx = i
# 目当ての要素を先頭に、残りはどうでもよし
a = A.pop(idx)
ans = [a] + A
print(*ans)
