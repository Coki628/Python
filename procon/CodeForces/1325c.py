# -*- coding: utf-8 -*-

"""
参考：誰かのツイート
・自力ならず。
・木、グラフ、MEX
・実はすごく簡単に、ってやつ。今回こういうのばっかだったな。。
・どうやるかはソース内コメント参照。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
nodes = [[] for i in range(N)]
ptoi = {}
ans = [0] * (N-1)
for i in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)
    ptoi[min(a, b), max(a, b)] = i

# 端点が3つ以上あるか確認
leaves = []
for i in range(N):
    if len(nodes[i]) == 1:
        leaves.append(i)
        if len(leaves) == 3:
            break

ans = [0] * (N-1)
if len(leaves) < 3:
    # 端点が2つだけ(一直線のグラフ)の場合は、MEX最大値はどうやってもN-2なので適当に
    for i in range(N-1):
        ans[i] = i
else:
    # 3つ以上あれば、それらの端点に0,1,2を割り当てて、あとは適当に
    edges = set()
    for i, u in enumerate(leaves):
        v = nodes[u][0]
        idx = ptoi[min(u, v), max(u, v)]
        edges.add(idx)
        ans[idx] = i
    cnt = 3
    for i in range(N-1):
        if i in edges:
            continue
        ans[i] = cnt
        cnt += 1
[print(a) for a in ans]
