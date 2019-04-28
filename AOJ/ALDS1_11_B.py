# -*- coding: utf-8 -*-

"""
・DFS
・再帰
"""

import sys
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

ans=[[0]*2 for i in range(N)]
visited=[False]*N
time=0
def rec(u):
    if visited[u]:
        return
    visited[u]=True
    global time
    time+=1
    ans[u][0]=time
    for v in nodes[u]:
        rec(v)
    time+=1
    ans[u][1]=time

for i in range(N):
    rec(i)
for i in range(N):
    d,f=ans[i]
    print(i+1, d, f)
