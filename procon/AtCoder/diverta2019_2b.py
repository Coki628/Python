# -*- coding: utf-8 -*-

"""
・300点なんとか自力AC(コンテスト終了2分半後…)
・2点間の距離と方向を一意に保持する方法が正しくなくて詰まってた。
　絶対値つけて、傾きが逆も考慮してやっと通った。
・傾き = Yの増加量 / Xの増加量
"""

import sys
from itertools import combinations
from collections import Counter

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

N=INT()

# 例外処理
if N==1:
    print(1)
    exit()

xy=[]
for i in range(N):
    x,y=MAP()
    xy.append((x, y))

C=Counter()
for comb in combinations(xy, 2):
    a,b=comb
    if a[0]-b[0]==0:
        key=INF
    else:
        key=(a[1]-b[1])/(a[0]-b[0])
    # 傾き、X距離、Y距離をキーとする
    C[(key, abs(a[0]-b[0]), abs(a[1]-b[1]))]+=1

mx=max(C.values())
print(N-mx)
