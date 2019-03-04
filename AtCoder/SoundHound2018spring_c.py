# -*- coding: utf-8 -*-

"""
・幅優先(WA)
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul, xor
from copy import copy, deepcopy
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
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

H,W=MAP()

# 右左上下
directions = [(0,1),(0,-1),(1,0),(-1,0)]
# 四方に一回り大きいグリッドを作る
grid = list2d(H+2, W+2, '*')
for i in range(1,H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

visited = list2d(H+2, W+2, False)

def bfs(start):
    que = deque()
    # スタート位置
    que.append([start,0])
    # 奇数カウント
    odd = 0
    # 偶数カウント
    even = 0
    while len(que):
        cur, dist = que.popleft()
        # 見終わった場所はもうやらない
        if visited[cur[0]][cur[1]]:
            continue
        # 訪問済にする
        visited[cur[0]][cur[1]] = True
        # 今のマスのカウント追加
        if dist%2==0:
            even += 1
        else:
            odd += 1
        # 4方向見る
        for direction in directions:
            # 上下左右に1つずつ動かした座標
            nxt = list(map(lambda x,y: x+y, cur, direction))
            # 壁はスキップ
            if grid[nxt[0]][nxt[1]] == '*':
                continue
            # キューに次のマスを足す
            que.append((nxt, dist+1))
    # 今回調べた範囲で得られる良い方のカウントを返却
    return max(odd, even)

cnt = 0
for i in range(1, H+1):
    for j in range(1, W+1):
        # 全てのマスについて調べる
        if grid[i][j]!='*' and not visited[i][j]:
            cnt += bfs((i, j))
print(cnt)
