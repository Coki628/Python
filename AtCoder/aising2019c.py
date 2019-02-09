# -*- coding: utf-8 -*-

"""
・幅優先
・自力AC
・python1.1秒、pypy0.6秒くらい。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

H, W = MAP()

# 右左上下
directions = [(0,1),(0,-1),(1,0),(-1,0)]
# 四方に一回り大きいグリッドを作る
grid = [['*'] * (W+2) for i in range(H+2)]
for i in range(1,H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

visited = list2d(H+2, W+2, False)

def bfs(start):
    que = deque()
    # スタート位置
    que.append(start)
    # 白マス通過カウント
    w_cnt = 0
    # 黒マス通過カウント
    b_cnt = 0
    while len(que):
        cur = que.popleft()
        # 見終わった場所はもうやらない
        if visited[cur[0]][cur[1]]:
            continue
        # 訪問済にする
        visited[cur[0]][cur[1]] = True
        # 今のマスのカウント追加
        if grid[cur[0]][cur[1]] == '.':
            w_cnt += 1
        elif grid[cur[0]][cur[1]] == '#':
            b_cnt += 1
        # 4方向見る
        for direction in directions:
            # 上下左右に1つずつ動かした座標
            nxt = list(map(lambda x,y: x+y, cur, direction))
            # 壁か今のマスと同じ色はスキップ
            if grid[nxt[0]][nxt[1]] == '*' or grid[cur[0]][cur[1]] == grid[nxt[0]][nxt[1]]:
                continue
            # キューに次のマスを足す
            que.append(nxt)
    # 今回調べた範囲で得られるカウントを返却
    return w_cnt * b_cnt

cnt = 0
for i in range(H+2):
    for j in range(W+2):
        # 全ての黒マスについて調べる
        if grid[i][j] == '#' and not visited[i][j]:
            cnt += bfs((i, j))
print(cnt)
