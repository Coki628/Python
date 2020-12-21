# -*- coding: utf-8 -*-

"""
・400点自力AC！
・回文行列
・似たようなのCFでやったことあってよかった。(1118c)
"""

import sys
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

H,W=MAP()
S=''
for i in range(H):
    S+=input()

C=Counter(S)
cnt1=cnt2=0
# 縦横とも偶数
if H%2==0 and W%2==0:
    for k, v in C.items():
        if v%4!=0:
            No()
            exit()
    # 全て4つ組
    Yes()
# 縦偶数
elif H%2==0:
    for k, v in C.items():
        if v%4!=0:
            if v%2==0:
                cnt2+=1
            else:
                No()
                exit()
    # 2つ組がH/2以下で残りが4つ組
    if cnt2<=H//2:
        Yes()
    else:
        No()
# 横偶数
elif W%2==0:
    for k, v in C.items():
        if v%4!=0:
            if v%2==0:
                cnt2+=1
            else:
                No()
                exit()
    # 2つ組がW/2以下で残りが4つ組
    if cnt2<=W//2:
        Yes()
    else:
        No()
# 両方奇数
else:
    for k, v in C.items():
        if v%4!=0:
            if v%2==0:
                cnt2+=1
            elif v%3==0:
                cnt1+=1
                cnt2+=1
            else:
                cnt1+=1
    # 1つ組1つ、2つ組が(H+W)/2-1以下で残りが4つ組
    if cnt1==1 and cnt2<=(H+W)//2-1:
        Yes()
    else:
        No()
