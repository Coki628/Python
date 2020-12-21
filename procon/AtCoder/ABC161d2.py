# -*- coding: utf-8 -*-

"""
・終了後20分くらいで自力AC。。
・グラフに帰着、BFS応用
・各数値を頂点にして、桁の少ない方から順に数値がK個集まるまでBFSしていく。
・こういうBFSに帰着させるやつ、最近こどふぉでもやったのに、中々気付けなかった。。
"""

import sys
from itertools import accumulate

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

def bfs():
    from collections import deque

    que = deque()
    cnt = 1
    memo = {}
    for i in range(1, 10):
        que.append(str(i))
        memo[cnt] = str(i)
        cnt += 1
    while que:
        x = que.popleft()
        last = int(x[-1])
        if last != 0:
            que.append(x+str(last-1))
            memo[cnt] = x + str(last-1)
            cnt += 1
        que.append(x+str(last))
        memo[cnt] = x + str(last)
        cnt += 1
        if last != 9:
            que.append(x+str(last+1))
            memo[cnt] = x + str(last+1)
            cnt += 1
        if cnt > K:
            return memo

K = INT()

res = bfs()
print(res[K])
