"""
・ここまで自力AC！
・木上のゲーム、木の直径
・題意のゲームは、アリスがボブを追いかけて捕まえられるか、ボブが永遠に逃げ切れるか、
　みたいなものになる。
・重要な点は3つあって、1つはボブがアリスの守備範囲を跨ぎ越すことができるかどうか。
　これができないと、アリスは少しずつボブを追い詰めて必ず捕まえられる。
　これは2人の守備範囲の比較でチェックできる。
・2つ目は、跨ぎ越すことができるとしても、初手でアリスがボブに届くなら問答無用でアリスは勝つ。
　これはアリスからボブにDFSなりで最短距離を取ればチェックできる。
・3つ目は、跨ぎ越せるくらいボブの守備範囲が広くても、そもそも木の領域が狭かったら意味がない。
　これは木の直径がDFS2回で取れるので、それでチェックでできる。
・これらの条件が揃った時だけ、ボブが勝つ。
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
INF = 10**19
MOD = 10**9 + 7

def dfs(nodes, src):

    N = len(nodes)
    stack = [(src, -1)]
    dist = [INF] * N
    dist[src] = 0
    while stack:
        u, prev = stack.pop()
        for v in nodes[u]:
            if v != prev:
                dist[v] = dist[u] + 1
                stack.append((v, u))
    return dist

for _ in range(INT()):
    N, a, b, da, db = MAP()
    a -= 1; b -= 1
    nodes = [[] for i in range(N)]
    for i in range(N-1):
        x, y = MAP()
        x -= 1; y -= 1
        nodes[x].append(y)
        nodes[y].append(x)

    dist = dfs(nodes, a)[b]
    if da*2+1 > db or dist <= da:
        print('Alice')
        continue

    res1 = dfs(nodes, 0)
    t = res1.index(max(res1))
    res2 = dfs(nodes, t)
    mx = max(res2)
    if da*2+1 > mx:
        print('Alice')
    else:
        print('Bob')
