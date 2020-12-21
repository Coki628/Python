# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/titia_til/status/1205837292767567873
・自力ならず。問題読み違えてた。。
・a,bを通って行けるペア、じゃなくて、a,bを通らないと行けないペア、だった。
・10万グラフ系
・頂点を1つないことにしたグラフでDFSする。
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

def dfs(start, avoid):
    res = [0] * N
    stack = [start]
    while stack:
        u = stack.pop()
        res[u] = 1
        for v in nodes[u]:
            if not res[v] and v != avoid:
                stack.append(v)
    return res

for _ in range(INT()):
    N, M, a, b = MAP()
    a -= 1; b -= 1
    nodes = [[] for i in range(N)]
    for i in range(M):
        x, y = MAP()
        x -= 1; y -= 1
        nodes[x].append(y)
        nodes[y].append(x)
    # aからbなしで行ける頂点
    froma = dfs(a, b)
    # bからaなしで行ける頂点
    fromb = dfs(b, a)
    aonly = bonly = 0
    for i in range(N):
        if i != a and i != b:
            # aからしか行けない
            if froma[i] and not fromb[i]:
                aonly += 1
            # bからしか行けない
            if fromb[i] and not froma[i]:
                bonly += 1
    print(aonly * bonly)
