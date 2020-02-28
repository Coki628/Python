# -*- coding: utf-8 -*-

"""
・木の走査、優先度付きキュー
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

# スタックDFSをちょっと変形して優先度付きキューでやる
def check(N, nodes):
    from heapq import heappush, heappop

    que = [(0, -1)]
    res = [0] * N
    cnt = 0
    while que:
        u, prev = heappop(que)
        res[cnt] = u + 1
        for v in nodes[u]:
            if v == prev:
                continue
            # 頂点番号の若い頂点を優先するようにする
            heappush(que, (v, u))
        cnt += 1
    return res

N = INT()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

res = check(N, nodes)
print(*res)
