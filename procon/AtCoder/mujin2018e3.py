# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/mujin-pc-2018/editorial.pdf
　　　https://betrue12.hateblo.jp/entry/2018/08/05/032043
・久々のグリッドダイクストラ
・待ち時間をO(1)で探索できるように前計算をやる。
・O(NMlogNM)=100万*20くらい=2000万がpypyでもTLE…。
・C++で同じの書いてAC0.3秒くらい。
・lambdaんとこが遅いって話になったから、変えてみたけどこれは無理だった。
　ちょっと速くはなったぽいけど。
・よく考えたら4方向分があるから2000万じゃなくて8000万相当か、そりゃきついか。
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

# あと何秒で特定の方向に動けるかを準備しておく
prep={'L':[-1]*K,'R':[-1]*K,'U':[-1]*K,'D':[-1]*K}
lcnt=rcnt=ucnt=dcnt=0
# 循環するので*2から始めて逆順でやる
for i in range(K*2-1,-1,-1):
    if d[i%K]=='L':
        lcnt=0
    if d[i%K]=='R':
        rcnt=0
    if d[i%K]=='U':
        ucnt=0
    if d[i%K]=='D':
        dcnt=0
    if i<K:
        prep['L'][i]=lcnt
        prep['R'][i]=rcnt
        prep['U'][i]=ucnt
        prep['D'][i]=dcnt
    lcnt+=1
    rcnt+=1
    ucnt+=1
    dcnt+=1

directions=[]
# 行ける方向だけ詰める
if d.count('R'):
    directions.append((0,1,'R'))
if d.count('L'):
    directions.append((0,-1,'L'))
if d.count('D'):
    directions.append((1,0,'D'))
if d.count('U'):
    # 高さはupがマイナスなので注意
    directions.append((-1,0,'U'))

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

def dijkstra(start):
    # 移動時間メモ
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
            nxt_t = cur_t
            nxt_x = cur_x + x
            nxt_y = cur_y + y
            # 壁はスキップ
            if grid[nxt_x][nxt_y] == '#':
                continue
            # 現在時間+1+待ち時間を移動先に記録
            if grid[nxt_x][nxt_y] == '.':
                stay=prep[k][cur_t%K]
                if stay==-1: continue
                dist = min(memo[cur_x][cur_y]+1+stay, memo[nxt_x][nxt_y])
                if not visited[nxt_x][nxt_y]:
                    memo[nxt_x][nxt_y] = dist
                    # 現在の経過時間をキューの優先度として、早い方から先に処理するようにする
                    nxt_t = dist
                    heappush(que, (nxt_t, nxt_x, nxt_y))
            # Gはキューに詰めないで記録のみ
            elif grid[nxt_x][nxt_y] == 'G':
                stay=prep[k][cur_t%K]
                if stay==-1: continue
                dist = min(memo[cur_x][cur_y]+1+stay, memo[nxt_x][nxt_y])
                memo[nxt_x][nxt_y] = dist
    # ゴール地点への最短時間を返却
    return memo[goal[1]][goal[2]]

ans=dijkstra(start)
if ans==INF:
    print(-1)
else:
    print(ans)
