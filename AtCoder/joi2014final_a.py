# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・いもす法
・計算量10万のO(N)でpythonAC0.4秒。
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

N, M = MAP()
P = [p-1 for p in LIST()]
ABC = []
for _ in range(N-1):
    a, b, c = MAP()
    ABC.append((a, b, c))

# 各鉄道を何回通るかをカウント
imos = [0] * N
for i in range(M-1):
    p1, p2 = P[i], P[i+1]
    if p2 < p1:
        p1, p2 = p2, p1
    imos[p1] += 1
    imos[p2] -= 1
imos = list(accumulate(imos))

ans = 0
for i, (a, b, c) in enumerate(ABC):
    cnt = imos[i]
    # 回数が分かっているので、どちらが得か比較できる
    cost = min(a*cnt, b*cnt+c)
    ans += cost
print(ans)
