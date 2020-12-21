# -*- coding: utf-8 -*-

"""
・DFSで木の走査
・参戦中は頂点1から順番に埋めたけど、こっちのがいいと思う。
・定形のDFSの動きに枝葉をつけただけな感じにできるから、実装しやすいしバグも生みにくそう。
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

# DFSで木の走査(開始位置はどこでもいい)
stack = [(0, -1)]
mx = 0
ans = [0] * (N-1)
while stack:
    u, prev = stack.pop()
    color = 0
    # 自分の子ノードを順番に塗り進めていく
    for v in nodes[u]:
        if v == prev:
            continue
        stack.append((v, u))
        color += 1
        a, b = min(u, prev), max(u, prev)
        # 親-自分間で使用済の色だったら1つ進める
        if a != -1 and color == edges[(a, b)][1]:
            color += 1
        a, b = min(u, v), max(u, v)
        edges[(a, b)][1] = color
        i = edges[(a, b)][0]
        ans[i] = str(color)
    # 各ノードから使った色数の最大を取る
    mx = max(mx, color)

print(mx)
print('\n'.join(ans))
