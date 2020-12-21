# -*- coding: utf-8 -*-

"""
・久々のグリッドダイクストラ
・まあTLEだけど。
・意外と方針は合ってた。待ち時間出しの下準備に何をやるかが出なかった。
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

N,M,K=MAP()
d=input()
# d2=defaultdict(list)
# for i,v in enumerate(d):
#     d2[v].append(i)

directions=[]
if d.count('R'):
    directions.append((0,1,'R'))
if d.count('L'):
    directions.append((0,-1,'L'))
if d.count('D'):
    directions.append((1,0,'D'))
if d.count('U'):
    # 高さはupがマイナスなので注意
    directions.append((-1,0,'U'))
# d2=''
# for i in range(2000000):
#     d2+=d[i%K]

start=None
goal=None

grid = list2d(N+2, M+2, '#')
for i in range(1,N+1):
    row = list(input())
    for j in range(1, M+1):
        grid[i][j] = row[j-1]
        if grid[i][j]=='S':
            start=(0,i,j)
        elif grid[i][j]=='G':
            goal=(INF,i,j)

def serach(s, t):
    t%=K
    res=d[t:].find(s)
    # 後ろに見つからなかったら先頭から再検索
    if res==-1:
        res=K-t
        res+=d.find(s)
    return res

def dijkstra(start):
    # 移動距離メモ
    memo = [[INF] * (M+2) for i in range(N+2)]
    # 訪問済メモ
    visited = [[False] * (M+2) for i in range(N+2)]
    # スタート位置
    que = [start]
    memo[start[1]][start[2]] = 0
 
    while len(que) != 0:
        cur_t, cur_x, cur_y = heappop(que)
        # 見終わった場所はもうやらない
        if visited[cur_x][cur_y]:
            continue
        # 訪問済
        visited[cur_x][cur_y] = True
        # 4方向見る
        for direction in directions:
            x, y, k, = direction
            direction = [0,x,y]
            nxt_t, nxt_x, nxt_y = map(lambda x,y: x+y, (cur_t, cur_x, cur_y), direction)
            # 壁はスキップ
            if grid[nxt_x][nxt_y] == '#':
                continue
            # 現在時間+1+待ち時間を移動先に記録
            if grid[nxt_x][nxt_y] == '.':
                stay=serach(k,cur_t)
                if stay==-1: continue
                dist = min(memo[cur_x][cur_y]+1+stay, memo[nxt_x][nxt_y])
                if not visited[nxt_x][nxt_y]:
                    memo[nxt_x][nxt_y] = dist
                    # 現在の経過時間をキューの優先度として、早い方から先に処理するようにする
                    nxt_t = dist
                    heappush(que, (nxt_t, nxt_x, nxt_y))
            # SかGはキューに詰めないで記録のみ
            else:
                stay=serach(k,cur_t)
                if stay==-1: continue
                dist = min(memo[cur_x][cur_y]+1+stay, memo[nxt_x][nxt_y])
                memo[nxt_x][nxt_y] = dist
 
    return memo[goal[1]][goal[2]]

ans=dijkstra(start)
if ans==INF:
    print(-1)
else:
    print(ans)
