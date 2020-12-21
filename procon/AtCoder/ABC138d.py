# -*- coding: utf-8 -*-

"""
・400点自力AC
・前計算して木を走査
・計算量はN+Qで両方20万なんだけど、以外とギリギリで
　python1.6秒pypyTLE、、やっぱ再帰が弱いのか。。
"""

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

N, Q = MAP()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    nodes[a-1].append(b-1)

ans = [0] * N
for i in range(Q):
    p, x = MAP()
    ans[p-1] += x

def dfs(u, prev, cnt):
    ans[u] += cnt
    for v in nodes[u]:
        # 有向の木だから実はこの逆走チェックはいらなかった
        if v != prev:
            dfs(v, u, ans[u])
dfs(0, -1, 0)

print(*ans)
