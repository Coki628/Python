# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-4、なんとか自力AC
・何かを決めるとGreedy
・危なかった。。最初考えた貪欲がWAですげー詰まった。
・女子が決まれば優先したい男子は一意に定まる、っていう概要は覚えてただけに、
　もっとさくっと決めれても良かったよなー。
"""

import sys
from itertools import combinations

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

ans = 0
# 女子の組み合わせ全通り
for comb in combinations(range(N), P):
    girls = set(comb)
    # 男子毎にポイントをつける
    boys = [0] * M
    for x, y, z in XYZ:
        # ペアの女子が来てくれるならポイント追加
        if x in girls:
            boys[y] += z
    # 今回のポイントに応じて、ソートして優先度確定
    boys.sort(reverse=True)
    ans = max(ans, sum(boys[:Q]))
print(ans)
