# -*- coding: utf-8 -*-

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
        if grid[i][j]=='#':
            que.append((i, j))

cnt=0
while len(que):
    ln=len(que)
    for i in range(ln):
        cur=que.popleft()
        # 4方向見る
        for direction in directions:
            nxt = list(map(lambda x,y: x+y, cur, direction))
            # 壁と黒はスキップ
            if grid[nxt[0]][nxt[1]]=='*' or grid[nxt[0]][nxt[1]]=='#':
                continue
            # 白を埋める
            if grid[nxt[0]][nxt[1]]=='.':
                grid[nxt[0]][nxt[1]]='#'
                # nxtの遷移先に白があったら詰める
                if (grid[nxt[0]-1][nxt[1]]=='.'
                        or grid[nxt[0]+1][nxt[1]]=='.'
                        or grid[nxt[0]][nxt[1]-1]=='.'
                        or grid[nxt[0]][nxt[1]+1]=='.'):
                    que.append(nxt)
    cnt+=1
print(cnt)
