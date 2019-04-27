# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc084/editorial.pdf
　　　https://www.youtube.com/watch?v=EwUlr0HXFDc&t=1593s
　　　https://atcoder.jp/contests/arc084/submissions/5108925
・桁和系
・01BFS
・*10したらコスト0、+1したらコスト1という辺を張ったグラフと考える。
"""

import sys
from collections import deque

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

K=INT()

que=deque()
que.append((1, 1))
costs=[-1]*K

while len(que):
    cur,cost=que.popleft()
    if costs[cur]==-1:
        costs[cur]=cost
        # コストのかからない*10は前に入れて優先して使えるようにする
        que.appendleft((cur*10%K, cost))
        # コストのかかる+1は後ろに入れて後回しにする
        que.append(((cur+1)%K, cost+1))
print(costs[0])
