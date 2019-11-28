# -*- coding: utf-8 -*-

"""
・DFSで木の走査
・参戦中は頂点1から順番に埋めたけど、こっちのがいいと思う。
・定形のDFSの動きに枝葉をつけただけな感じにできるから、実装しやすいしバグも生みにくそう。
・ちょっと気になったから色々再整理。
・親との辺の色を一緒に持たせたらすっきり。
・速度も改善。10万頂点でpythonAC0.9→0.7秒、pypyAC0.6→0.5秒。
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
    edges[(a, b)] = i

# DFSで木の走査(開始位置はどこでもいい)
# (自分, 親, 親との辺の色)
stack = [(0, -1, -1)]
mx = 0
ans = [0] * (N-1)
while stack:
    u, prev, used = stack.pop()
    color = 0
    # 自分の子ノードを順番に塗り進めていく
    for v in nodes[u]:
        if v == prev:
            continue
        color += 1
        # 親-自分間で使用済の色だったら1つ進める
        if color == used:
            color += 1
        a, b = min(u, v), max(u, v)
        i = edges[(a, b)]
        ans[i] = str(color)
        stack.append((v, u, color))
    # 各ノードから使った色数の最大を取る
    mx = max(mx, color)

print(mx)
print('\n'.join(ans))
