# -*- coding: utf-8 -*-

"""
・自力AC！
・昔のだから点数ないけど、ARCのCだから500~800点相当くらい？
・グリッド斜め走破のループ添字が結構悩んだ。
　次あったらこれ使ってさくっと書こう。
"""

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

def build_grid(H, W, offset=0, init='*'):
    grid = list2d(H+offset*2, W+offset*2, init)
    for i in range(offset, H+offset):
        row = list(input())
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

# 四方に9回り大きいグリッドを作る
H=W=19
grid=build_grid(H, W, 9, '*')

# まずは手番の整合性チェック
brcnt=whcnt=0
for i in range(9, H+9):
    for j in range(9, W+9):
        if grid[i][j]=='o':
            brcnt+=1
        elif grid[i][j]=='x':
            whcnt+=1
if not (whcnt<=brcnt<=whcnt+1):
    NO()
    exit()

# 5連続以上があるかのチェック
def check(grid):
    # 横方向
    for i in range(9, H+9):
        brcnt=0
        whcnt=0
        for j in range(9, W+9):
            if grid[i][j]=='o':
                brcnt+=1
                whcnt=0
            elif grid[i][j]=='x':
                brcnt=0
                whcnt+=1
            else:
                brcnt=0
                whcnt=0
            if 5<=brcnt or 5<=whcnt:
                return
            j+=1
    # 縦方向
    for i in range(9, W+9):
        brcnt=0
        whcnt=0
        for j in range(9, H+9):
            if grid[j][i]=='o':
                brcnt+=1
                whcnt=0
            elif grid[j][i]=='x':
                brcnt=0
                whcnt+=1
            else:
                brcnt=0
                whcnt=0
            if 5<=brcnt or 5<=whcnt:
                return
            j+=1
    # 斜め方向
    for i in range(19):
        # 左上→右下
        for j in range(19):
            if grid[i+j][-i+j+18]=='o':
                brcnt+=1
                whcnt=0
            elif grid[i+j][-i+j+18]=='x':
                brcnt=0
                whcnt+=1
            else:
                brcnt=0
                whcnt=0
            if 5<=brcnt or 5<=whcnt:
                return
            j+=1
        for j in range(18):
            if grid[i+j+1][-i+j+18]=='o':
                brcnt+=1
                whcnt=0
            elif grid[i+j+1][-i+j+18]=='x':
                brcnt=0
                whcnt+=1
            else:
                brcnt=0
                whcnt=0
            if 5<=brcnt or 5<=whcnt:
                return
            j+=1
    for i in range(19):
        # 右上→左下
        for j in range(19):
            if grid[i+j][i-j+18]=='o':
                brcnt+=1
                whcnt=0
            elif grid[i+j][i-j+18]=='x':
                brcnt=0
                whcnt+=1
            else:
                brcnt=0
                whcnt=0
            if 5<=brcnt or 5<=whcnt:
                return
            j+=1
        for j in range(18):
            if grid[i+j+1][i-j+18]=='o':
                brcnt+=1
                whcnt=0
            elif grid[i+j+1][i-j+18]=='x':
                brcnt=0
                whcnt+=1
            else:
                brcnt=0
                whcnt=0
            if 5<=brcnt or 5<=whcnt:
                return
            j+=1
    YES()
    exit()

check(grid)
for i in range(9, H+9):
    for j in range(9, W+9):
        # 1ターン前を全通り試して、終了状態じゃないものがあればOK
        if ((grid[i][j]=='o' and brcnt==whcnt+1)
                or (grid[i][j]=='x' and brcnt==whcnt)):
            # 該当箇所を操作前の状態にする
            prev=grid[i][j]
            grid[i][j]='.'
            check(grid)
            # チェック後には操作後の状態に戻しておく
            grid[i][j]=prev
NO()
