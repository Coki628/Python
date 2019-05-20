# -*- coding: utf-8 -*-

"""
・400点自力AC
・木の探索
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

N=INT()
nodes=[[] for i in range(N)]
for i in range(N-1):
    u,v,w=MAP()
    nodes[u-1].append((v-1, w))
    nodes[v-1].append((u-1, w))

ans=[-1]*N
def rec(cur, color):
    if ans[cur]!=-1:
        return
    ans[cur]=color
    for node in nodes[cur]:
        nxt,w=node
        if w%2==0:
            rec(nxt, color)
        else:
            rec(nxt, 1-color)
    return

rec(0, 0)

for i in range(N):
    print(ans[i])
