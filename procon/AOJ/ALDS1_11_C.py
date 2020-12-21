# -*- coding: utf-8 -*-

"""
・BFS
・キュー
"""

import sys
from collections import deque

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

N=INT()
nodes=[[] for i in range(N)]
for i in range(N):
    l=LIST()
    u=l[0]
    l=l[2:]
    for v in l:
        nodes[u-1].append(v-1)
    nodes[u-1].sort()

que=deque()
que.append((0, 0))
costs=[-1]*N
while len(que):
    cost,node=que.popleft()
    if costs[node]==-1:
        costs[node]=cost
        for v in nodes[node]:
            que.append((cost+1, v))

for i in range(N):
    print(i+1, costs[i])
