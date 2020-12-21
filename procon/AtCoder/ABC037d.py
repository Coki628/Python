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
# 四方に一回り大きいグリッドを作る
grid = list2d(H+2, W+2, INF)
directions = [(0,1),(0,-1),(1,0),(-1,0)]
for i in range(1, H+1):
    row = LIST()
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

que=deque()
done=list2d(H+2, W+2, True)
send=list2d(H+2, W+2, True)
route=list2d(H+2, W+2, 0)
for i in range(1, H+1):
    for j in range(1, W+1):
        if (grid[i-1][j]>=grid[i][j]
                and grid[i+1][j]>=grid[i][j]
                and grid[i][j-1]>=grid[i][j]
                and grid[i][j+1]>=grid[i][j]):
            que.append((i, j))
            route[i][j]=1
        else:
            done[i][j]=False
            send[i][j]=False

cost=list2d(H+2, W+2, 0)
sm=list2d(H+2, W+2, 0)

ans=0
while len(que):
    h1,w1=que.popleft()

    # 4方向見る
    is_ok=True
    for h, w in directions:
        h2=h1+h
        w2=w1+w
        # もう遷移させていい場所かの確認
        if not done[h2][w2] and grid[h2][w2]<grid[h1][w1]:
            is_ok=False
    if is_ok:
        cnt=0
        for h, w in directions:
            h2=h1+h
            w2=w1+w
            if grid[h2][w2]!=INF and grid[h2][w2]>grid[h1][w1]:
                route[h2][w2]+=route[h1][w1]
                cost[h2][w2]+=cost[h1][w1]+1
                sm[h2][w2]+=sm[h1][w1]+cost[h1][w1]+1
                if not send[h2][w2]:
                    que.append((h2, w2))
                    send[h2][w2]=True
                cnt+=1
        done[h1][w1]=True
        if cnt==0:
            ans+=sm[h1][w1]
    else:
        send[h1][w1]=False

print(ans+H*W)

# なんで多くなるかは多分分かった。
# 1本→分かれて2本→また合流、みたいな時に、
# 2本の時の数え上げで、1本の時の分まで2重に数えちゃってる。
# でもどこが1本だったか、とかを後から識別するうまい方法が見つからないや。。

