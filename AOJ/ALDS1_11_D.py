# -*- coding: utf-8 -*-

"""
・連結成分
・BFSで訪問済にする時にグループ番号を付ける。
"""

import sys

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

N,M=MAP()
nodes=[[] for i in range(N)]
for i in range(M):
    s,t=MAP()
    nodes[s].append(t)
    nodes[t].append(s)

group=[-1]*N
def rec(u, root):
    if group[u]!=-1:
        return
    group[u]=root
    for v in nodes[u]:
        rec(v, root)

for i in range(N):
    rec(i, i)

Q=INT()
for _ in range(Q):
    s,t=MAP()
    if group[s]==group[t]:
        print('yes')
    else:
        print('no')
