# -*- coding: utf-8 -*-

"""
・実験用。実際にやったら規則性は明らかで、N=5辺りでもう分かる。
"""

import sys
from itertools import product, combinations

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
A = LIST()

A.sort()

def f(A, B):
    res = [0] * N
    cnt = 0
    # まず必要な操作回数の合計
    for i in range(N):
        if A[i] != B[i]:
            cnt += 1
    # 残り回数分追加して、次は1つ減る
    for i in range(N):
        if A[i] != B[i]:
            res[i] = cnt
            cnt -= 1
    return res

sm = [0] * N
for comb in combinations(product([0, 1], repeat=N), 2):
    res = f(*comb)
    for i in range(N):
        sm[i] += res[i]
print(sm)
ans = 0
for i in range(N):
   ans += sm[i] * A[i]
# print(ans * 2)
