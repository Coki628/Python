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

N, M = MAP()
nodes =[[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    nodes[a-1].append(b-1)
    nodes[b-1].append(a-1)
C = [c-1 for c in LIST()]

cnts = [0] * N
done = [False] * N
def rec(cur, prev):
    if done[cur] or len(nodes[cur]) > 2:
        return
    for nxt in nodes[cur]:
        if nxt != prev:
            if len(nodes[nxt]) > 2:
                cnts[nxt] += 1
            else:
                rec(nxt, cur)
    done[cur] = True

for c in C:
    rec(c, -1)
