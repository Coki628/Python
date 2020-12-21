# -*- coding: utf-8 -*-

"""
・BFS、偶奇
・問題読み違えて難しいこと考えてたけど、結局は距離取って偶奇で振り分ければOK。
・20万の入出力が最初2秒TLEで、最後にまとめて出力するように変えたら無事pypyAC0.7秒。かなり変わるな。
"""

import sys
from collections import deque

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bfs(nodes, src):
    N = len(nodes)
    que = deque([(src, 0)])
    dist = [INF] * N
    dist[src] = 0
    while que:
        u, c = que.popleft()
        for v in nodes[u]:
            if dist[v] != INF:
                continue
            dist[v] = c + 1
            que.append((v, c+1))
    return dist

ans = []
for _ in range(INT()):
    N, M = MAP()
    nodes = [[] for i in range(N)]
    for _ in range(M):
        a, b = MAP()
        a -= 1; b -= 1
        nodes[a].append(b)
        nodes[b].append(a)
    
    res = bfs(nodes, 0)
    ans1 = []
    ans2 = []
    for u, c in enumerate(res):
        if c % 2 == 0:
            ans1.append(str(u+1))
        else:
            ans2.append(str(u+1))
    if len(ans1) <= len(ans2):
        ans.append(str(len(ans1)))
        ans.append(' '.join(ans1))
    else:
        ans.append(str(len(ans2)))
        ans.append(' '.join(ans2))
print('\n'.join(ans))
