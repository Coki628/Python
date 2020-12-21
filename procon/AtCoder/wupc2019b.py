# -*- coding: utf-8 -*-

import sys

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

# 縦横1マスのみ
if H==1 and W==1:
    num=INT()
    if num==0:
        print('Yes', 0)
    elif num==5:
        print('Yes', 1)
    else:
        No()
    exit()

# 縦横どちらか1マスなら、どの5を残すかで最適を探す
if H==1 or W==1:
    if H==1:
        grid=LIST()
    elif W==1:
        grid=[INT() for i in range(H)]
    # 全て0の場合は何もしなくていい
    if all(grid[i]==0 for i in range(len(grid))):
        print('Yes', 0)
        exit()
    idx5=[]
    for i in range(len(grid)):
        if grid[i]==5:
            idx5.append(i)
    # 5が1つもなければNG
    if not len(idx5):
        No()
        exit()
    ans=INF
    for i in idx5:
        # グリッドを5より手前と後で分ける
        grid1=grid[:i]
        grid2=grid[i+1:]
        mx=[0]*2
        for j in range(len(grid1)):
            mx[0]=max(mx[0], max(grid1))
        for j in range(len(grid2)):
            mx[1]=max(mx[1], max(grid2))
        cnt=0
        for j in range(2):
            # 5以下に下げるために必要な回数
            if 6<=mx[j]<=7:
                cnt+=1
            elif mx[j]==8:
                cnt+=2
            elif mx[j]==9:
                cnt+=3
        # 最後に5以下の全てを0にする1回
        cnt+=1
        ans=min(ans, cnt)
    print('Yes', ans)
    exit()

# 縦横どちらかでも2マス以上あれば、
# 好きな5を1つ残しながら他の全てを一度に変更できるので
# 5があるかと全体の最大値だけ確認できれば回数が分かる。
mx=0
has5=False
grid=[None for i in range(H)]
for i in range(H):
    grid[i]=LIST()
    mx=max(mx, max(grid[i]))
    if grid[i].count(5):
        has5=True

if mx==0:
    print('Yes', 0)
elif 1<=mx<=4:
    No()
elif mx==5:
    print('Yes', 1)
elif 6<=mx<=7 and has5:
    print('Yes', 2)
elif mx==8 and has5:
    print('Yes', 3)
elif mx==9 and has5:
    print('Yes', 4)
else:
    No()
