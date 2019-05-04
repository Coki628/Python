# -*- coding: utf-8 -*-

"""
・グリッドのBFS(久しぶり)
・H*W*4方向=400万は結構きつめではあるのものの、pypyでも通らないのは腑に落ちず。
・C++でほぼ同じの書いて通した。
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

H,W=MAP()

que = deque()
# 右左上下  
directions = [(0,1),(0,-1),(1,0),(-1,0)]
# 四方に一回り大きいグリッドを作る
grid = list2d(H+2, W+2, '*')
for i in range(1,H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]
        if grid[i][j] == '#':
            que.append((i, j, 0))

# 移動距離メモ
memo=[[-1]*(W+2) for i in range(H+2)]
while len(que):
    cur=que.popleft()
    if memo[cur[0]][cur[1]]!=-1:
        continue
    memo[cur[0]][cur[1]]=cur[2]
    # 4方向見る
    for direction in directions:
        x,y=map(lambda x,y: x+y, cur, direction)
        # 壁と黒はスキップ
        if grid[x][y]=='*' or grid[x][y]=='#':
            continue
        # 白を埋める
        if grid[x][y]=='.':
            grid[x][y]='#'
            que.append((x, y, cur[2]+1))
ans=0
for i in range(1, H+1):
    for j in range(1, W+1):
        ans=max(ans, memo[i][j])
print(ans)
