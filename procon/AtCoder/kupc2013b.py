# -*- coding: utf-8 -*-

"""
・全探索、累積和
・NとXが小さいので、N個のオリと0〜X匹のライオンのパターンを全探索できる。
"""

import sys
from itertools import product, accumulate

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

N, X, M = MAP()
LR = []
for i in range(M):
    l, r, s = MAP()
    LR.append((l, r, s))

# 各オリのライオンの数を全探索
mx = -1
ans = (-1,)
for prod in product(range(X+1), repeat=N):
    acc = [0] + list(accumulate(prod))
    for l, r, s in LR:
        if acc[r] - acc[l-1] != s:
            break
    else:
        # 条件を満たしたら、合計が最大か確認
        sm = sum(prod)
        if sm > mx:
            mx = sm
            ans = prod
print(*ans)
