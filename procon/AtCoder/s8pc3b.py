# -*- coding: utf-8 -*-

"""
・テトリスみたいなやつ
・実装問題。疲れた。
"""

import sys
from copy import deepcopy

def input(): return sys.stdin.readline().strip()
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

H,W,K=MAP()
orgn=[[0]*W for i in range(H)]
for i in range(H):
    nums=input()
    for j in range(W):
        orgn[i][j]=int(nums[j])

# ブロックが落ちる動き
def drop():
    for i in range(H-1, 0, -1):
        for j in range(W):
            if grid[i][j]==0 and grid[i-1][j]!=0:
                k=i
                while k<H and grid[k][j]==0:     
                    grid[k][j]=grid[k-1][j]
                    grid[k-1][j]=0
                    k+=1

# 揃ったところを消して合計を計算
def check(times):
    sm=0
    for i in range(H):
        st=0
        cnt=1
        for j in range(1, W):
            if grid[i][j-1]==grid[i][j]:
                cnt+=1
            else:
                if cnt>=K:
                    sm+=grid[i][j-1]*cnt
                    for k in range(st, j):
                        grid[i][k]=0
                cnt=1
                st=j
        else:
            if cnt>=K:
                sm+=grid[i][j]*cnt
                for k in range(st, j+1):
                    grid[i][k]=0
    return 2**times*sm

ans=0
for i in range(H):
    for j in range(W):
        # 元の状態は保持しておいて、全部試す
        grid=deepcopy(orgn)
        grid[i][j]=0    
        res=1
        score=times=0
        # 消えるところがある限りループ
        while res:
            drop()
            res=check(times)
            score+=res
            times+=1
        ans=max(ans, score)

print(ans)
