# -*- coding: utf-8 -*-

import sys
from collections import Counter

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

N,K=MAP()

# 循環の1周に何がどう含まれるかを予め作っておく
grid=[[0]*5 for i in range(11)]
for i in range(11):
    num=81+i*63
    for j in range(5):
        grid[i][j]=num%11
        num+=9

# 1循環中の要素数確認 → 全部5つずつ
# C=Counter()
# for i in range(11):
#     C.update(grid[i])

# 循環している回数*5
# 最後の2行は3*3の正方形を作れないので除く
ans=(N-2)//11*5
# 最後の1回は必要なところまでを数える
mod=(N-2)%11
for i in range(mod):
    for j in range(5):
        if grid[i][j]==K:
            ans+=1
print(ans)
