# -*- coding: utf-8 -*-

"""
・自分がやろうとしてた、完全グラフから引いていく方。
・最終形をウニグラフにする(ある頂点からだけは辺を引かない)って方針が分かれば、
　素直に1辺減らして1組増えるにできたんだね。。
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

N,K=MAP()

edges=[]
for i in range(N):
    for j in range(i+1, N):
        edges.append((i+1, j+1))

if K>(N-1)*(N-2)//2:
    print(-1)
    exit()
else:
    # 後ろから順に減らしていけば、最後には頂点1との辺だけが残るのでOK
    for i in range(K):
        edges.pop()

print(len(edges))
for u, v in edges:
    print(u, v)
