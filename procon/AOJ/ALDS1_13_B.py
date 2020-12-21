# -*- coding: utf-8 -*-

"""
・8パズル
・一応自力AC
・盤面の状態を頂点とみなしてBFS
・毎回いちいちkey<=>gird変換してるから遅いだろうなぁと思ったんだけど、
　出してみたらちゃんと通った。
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

N = 3
grid = list2d(N+2, N+2, '*')
for i in range(1, N+1):
    row = list(input().split())
    for j in range(1, N+1):
        grid[i][j] = row[j-1]
        if row[j-1] == '0':
            start = (i, j)

def get_key(grid):
    res = []
    for i in range(N+2):
        res.append(''.join(grid[i]))
    return ''.join(res)

def get_grid(key):
    grid = list2d(N+2, N+2, '')
    for i in range(N+2):
        for j in range(N+2):
            grid[i][j] = key[i*(N+2)+j]
    return grid

# 完成形の盤面
goal = '******123**456**780******'
key = get_key(grid)
que = deque()
que.append((key, 0, start[0], start[1]))
# その盤面に至る移動コスト
memo = {}
directions = ((1,0),(-1,0),(0,1),(0,-1))
while len(que):
    key, cost, h1, w1 = que.popleft()
    if key in memo:
        continue
    memo[key] = cost
    grid = get_grid(key)
    # 4方向見る
    for direction in directions:
        h2 = h1 + direction[0]
        w2 = w1 + direction[1]
        if grid[h2][w2] != '*':
            grid[h2][w2], grid[h1][w1] = grid[h1][w1], grid[h2][w2]
            que.append((get_key(grid), cost+1, h2, w2))
            grid[h2][w2], grid[h1][w1] = grid[h1][w1], grid[h2][w2]

print(memo[goal])
