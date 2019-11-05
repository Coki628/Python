# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-4
・何かを決めるとGreedy
・これはWA。。
"""

import sys
from itertools import combinations
from operator import itemgetter

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

N, M, P, Q, R = MAP()

XYZ = []
for i in range(R):
    x, y, z = MAP()
    XYZ.append((x-1, y-1, z))

# ポイント高い順ソート
XYZ.sort(key=itemgetter(2), reverse=True)

ans = 0
# 女子の組み合わせ全通り
for comb in combinations(range(N), P):
    girls = set(comb)
    boys = set()
    cnt = 0
    for x, y, z in XYZ:
        if x in girls:
            if y in boys:
                cnt += z
            # ポイントの高い組に絡む男子から優先して入れていく
            elif len(boys) < Q:
                cnt += z
                boys.add(y)
    ans = max(ans, cnt)
print(ans)
