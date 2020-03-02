# -*- coding: utf-8 -*-

"""
参考：http://www.utpc.jp/2012/slides/mori.pdf
・森の判定、閉路検出
・始まりが完全グラフなので、辺が多すぎる場合は最初に除外する。
・するとやる必要があるのは頂点がだいぶ少ない時に限られる。
・具体的には3~400くらいになるはずで、10万クエリに毎回BFSしても3~4000万くらいの
　計算量になると思うんだけど、pypyでもTLE。。。
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

def dfs(N, nodes, src):
    """ DFS(木、スタック、重みなし) """

    stack = [(src, -1, 0)]
    dist = [INF] * N
    while stack:
        u, prev, c = stack.pop()
        # 別の始点からの走査で確認済
        if prev == -1 and dist[u] != INF:
            return True
        # 閉路検出
        if dist[u] != INF:
            return False
        dist[u] = c
        for v in nodes[u]:
            if v != prev:
                stack.append((v, u, c+1))
    return True

def check():
    for i in range(N):
        if not dfs(N, nodes, i):
            return False
    return True

N, M = MAP()
cnt = N*(N-1) // 2
# 全クエリで減らしても辺数がN-1以下にならない
if cnt - M > N-1:
    [print('no') for i in range(M)]
    exit()
# 完全グラフの構築
nodes = [set() for i in range(N)]
for a in range(N):
    for b in range(a+1, N):
        nodes[a].add(b)
        nodes[b].add(a)

for _ in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    # 辺の増減
    if b in nodes[a]:
        nodes[a].remove(b)
        nodes[b].remove(a)
        cnt -= 1
    else:
        nodes[a].add(b)
        nodes[b].add(a)
        cnt += 1
    # 辺数がN-1以下なら判定する
    if cnt <= N-1:
        if check():
            print('yes')
        else:
            print('no')
    else:
        print('no')
