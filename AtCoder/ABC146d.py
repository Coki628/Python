# -*- coding: utf-8 -*-

"""
・400点自力AC、試行錯誤の末。
・色決めるwhileが余計に回ってた。
・でももっとシンプルにいけたと思う。
"""

import sys

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
nodes = [[] for i in range(N)]
edges = {}
for i in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)
    edges[(a, b)] = [i, -1]

mx = 0
used = [set() for i in range(N)]
ans = [0] * (N-1)
cnt = 0
for u in range(N):
    mx = max(mx, len(nodes[u]))
    color = 1
    for v in nodes[u]:
        a, b = min(u, v), max(u, v)
        if edges[(a, b)][1] == -1:
            while color in used[a] or color in used[b]:
                color += 1
            edges[(a, b)][1] = color
            used[a].add(color)
            used[b].add(color)
            i = edges[(a, b)][0]
            ans[i] = str(color)
            color += 1
            cnt += 1
            if cnt == N-1:
                break
    else:
        continue
    break
print(mx)
print('\n'.join(ans))
