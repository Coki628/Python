# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc001
・今日の一問から
・BFS、確かにこれでもできるな。。
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

A, B = MAP()

def bfs(start, goal):

    que = deque()
    que.append((0, start))

    memo = [-1] * 41
    while len(que):
        cost, cur = que.popleft()
        # 範囲外or訪問済はスキップ
        if not 0 <= cur <= 40 or memo[cur] != -1:
            continue
        memo[cur] = cost
        # 6通りの操作をキューに詰める
        que.append((cost+1, cur+1))
        que.append((cost+1, cur+5))
        que.append((cost+1, cur+10))
        que.append((cost+1, cur-1))
        que.append((cost+1, cur-5))
        que.append((cost+1, cur-10))
    
    return memo[goal]

print(bfs(A, B))
