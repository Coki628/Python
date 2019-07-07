# -*- coding: utf-8 -*-

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
nodes=[[] for i in range(N)]
for i in range(N-1):
    a,b=MAP()
    nodes[a-1].append(b-1)
    nodes[b-1].append(a-1)

visited=[False]*N
ans=[0]*N
def rec(u, cnt):
    if visited[u]:
        return
    visited[u]=True
    if cnt>1:
        ans[u]=max(cnt, K-2)
    else:
        ans[u]=1
    for v in nodes[u]:
        rec(v, cnt-1)

for i in range(N):
    if len(nodes[i])==1:
        rec(i, K)
        break

cnt=1
for i in range(N):
    cnt*=ans[i]
    cnt%=MOD
print(cnt)
