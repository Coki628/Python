# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc045
・自力ならず。。
・XOR系、DFS、木の走査
・XORが同値を打ち消すのをうまく活用できる。
・具体的には、根からのコストさえ持っていれば、共通する辺を打ち消して、ある2頂点間のコストを全部出せる。
・後はdictで通った各場所のコストを管理すれば、自分とペアになってちょうどXになる頂点がいくつあるかすぐ分かる。
・こういうXORの特性を活かして何かうまくやるやつ、見抜けるようになりたいなぁ。。
"""

import sys
from collections import defaultdict

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

N, X = MAP()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))

stack = [(0, -1, 0)]
d = defaultdict(int)
ans = 0
while stack:
    u, prev, cost = stack.pop()
    # 自分とペアになってちょうどXを作れるやつの数を答えに追加
    ans += d[cost^X]
    d[cost] += 1
    for v, c in nodes[u]:
        if v != prev:
            stack.append((v, u, cost^c))
print(ans)
