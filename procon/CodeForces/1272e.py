# -*- coding: utf-8 -*-

"""
・自力ならず。メモ化再帰でDFSやろうとしてダメだった。
・他の人のさらっと見たら、みんなBFSっぽいことしてたからその視点で考え直したら何とか自力実装AC。
・グラフに帰着、BFS
・正解は終点からBFS。終点となりうる場所はコスト1で全て確定するので、そこを全てBFSの始点にする。
・終点側からはどの頂点が入ってくるか分からないので、始点を決める最中に逆辺を張っておく。
"""

import sys
from collections import deque

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
A = LIST()

# まず終点になる場所が全てコスト1で確定して始点になる
que = deque()
nodes = [[] for i in range(N)]
for i, a in enumerate(A):
    if i + a < N:
        # 終点から辿ってこれるように逆辺を張る
        nodes[i+a].append(i)
        if a % 2 != A[i+a] % 2:
            que.append((i, 1))
            continue
    if i - a >= 0:
        nodes[i-a].append(i)
        if a % 2 != A[i-a] % 2:
            que.append((i, 1))

# 後は普通にBFSするだけ
ans = [-1] * N
while que:
    u, cost = que.popleft()
    if ans[u] != -1:
        continue
    ans[u] = cost
    for v in nodes[u]:
        que.append((v, cost+1))
print(*ans)
