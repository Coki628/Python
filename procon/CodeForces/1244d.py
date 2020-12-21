# -*- coding: utf-8 -*-

"""
・自力ACならず。RE。。
・どうやら原因は再帰が深すぎだったからぽい。
・setrecursionlimit使っても、10万まで潜るような深い再帰はしない方がいいみたい。
"""

import sys
from itertools import permutations

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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
C1 = LIST()
C2 = LIST()
C3 = LIST()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    a -= 1
    b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

start = 0
for i in range(N):
    if len(nodes[i]) >= 3:
        print(-1)
        exit()
    if len(nodes[i]) == 1:
        start = i

Cs = [[] for i in range(3)]
def rec(u, prev):
    Cs[0].append(C1[u])
    Cs[1].append(C2[u])
    Cs[2].append(C3[u])
    for v in nodes[u]:
        if v != prev:
            rec(v, u)
rec(start, -1)

mn = INF
p = []
for perm in permutations(range(3)):
    sm = 0
    for i in range(N):
        sm += Cs[perm[i%3]][i]
    if mn > sm:
        mn = min(mn, sm)
        p = perm

ans = [0] * N
def rec2(u, prev, cnt):
    ans[u] = p[cnt%3] + 1
    for v in nodes[u]:
        if v != prev:
            rec2(v, u, cnt+1)
rec2(start, -1, 0)

print(mn)
print(*ans)
