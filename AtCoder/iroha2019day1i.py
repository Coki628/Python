# -*- coding: utf-8 -*-

"""
・500点自力AC
・01BFS
・でも普通にダイクストラでも解けたみたい。
"""

import sys
from collections import deque

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

N,M,K=MAP()
nodes=[[] for i in range(N)]
for i in range(M):
    a,b,group=MAP()
    nodes[a-1].append((b-1, group))
    nodes[b-1].append((a-1, group))

def bfs01(start, startgroup):
    que=deque()
    cost=[INF]*N
    cost[0]=0
    cost[start]=0
    que.append((start, startgroup))
    while len(que):
        cur,curgroup=que.popleft()
        for nxt, nxtgroup in nodes[cur]:
            if cost[nxt]==INF:
                if curgroup==nxtgroup:
                    # コストのかからない同じグループへの移動は前に入れて優先して使えるようにする
                    cost[nxt]=cost[cur]
                    que.appendleft((nxt, nxtgroup))
                else:
                    # コストのかかる違うグループへの移動は後ろに入れて後回しにする
                    cost[nxt]=cost[cur]+1
                    que.append((nxt, nxtgroup))
    return cost[N-1]

ans=INF
for nxt, group in nodes[0]:
    # 最初の移動はどれもコスト0なので全通り試す
    ans=min(ans, bfs01(nxt, group))
if ans!=INF:
    print((ans+1)*K)
else:
    print(-1)
