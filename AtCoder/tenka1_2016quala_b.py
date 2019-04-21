# -*- coding: utf-8 -*-

"""
・300点自力AC！(結構辛口めな300点と思う)
・BFSで木を探索
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N,M=MAP()
nodes=[[] for i in range(N)]
for i in range(1, N):
    nodes[INT()].append(i)
need=[0]*N
for i in range(M):
    b,c=MAP()
    need[b]=c

sm=0
def rec(cur):
    global sm
    # 末端のなら必要な値を親に返す
    if not len(nodes[cur]):
        return need[cur]
    # 自分の子について戻り値を集める
    costs=[]
    for v in nodes[cur]:
        costs.append(rec(v))
    res=0
    # 根ならそのまま合計に足す
    if cur==0:
        for cost in costs:
            sm+=cost
    # 他は最小値を引いた値を合計に足していき、最小値の分は親に返す
    else:
        mn=min(costs)
        for cost in costs:
            sm+=cost-mn
        res=mn
    return res

rec(0)
print(sm)
