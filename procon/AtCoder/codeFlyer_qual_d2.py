# -*- coding: utf-8 -*-

"""
参考：https://betrue12.hateblo.jp/entry/2018/06/03/131350
・区分を分けて考える。
・2次元いもす
・2000^2=400万のループ最大6回=2400万はきついかなーと思ったけどpypyで普通に通った。
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

H,W=MAP()
N,M=MAP()

grid=[[None] for i in range(N)]
for i in range(N):
    grid[i]=list(input())

# まず2次元いもすでA領域を数える
H2=min(H, N*2)
W2=min(W, M*2)
# 動かさないのが1なので、縦横に動かす長さ+1
di=H2-N+1
dj=W2-M+1
imos=[[0]*(W2+1) for i in range(H2+1)]
for i in range(N):
    for j in range(M):
        if grid[i][j]=='#':
            imos[i][j]+=1
            imos[i+di][j]-=1
            imos[i][j+dj]-=1
            imos[i+di][j+dj]+=1
cnt=0
for i in range(H2):
    for j in range(W2):
        imos[i][j+1]+=imos[i][j]
for j in range(W2):
    for i in range(H2):
        imos[i+1][j]+=imos[i][j]
for i in range(H2):
    for j in range(W2):
        if imos[i][j]==0:
            cnt+=1
# 黒が1つもないケースを確認
allwhite=False
if cnt==H2*W2:
    allwhite=True

# 縦が長ければB領域を数える
if H>N*2:
    iswhitecol=[False]*W2
    colcnt=0
    for j in range(W2):
        for i in range(H2):
            if imos[i][j]!=0:
                break
        else:
            iswhitecol[j]=True
            colcnt+=1
    # いもすで先に数えた分を引いてから、列全体を足す
    cnt-=N*2*colcnt
    cnt+=H*colcnt

# 横が長ければC領域を数える
if W>M*2:
    iswhiterow=[False]*H2
    rowcnt=0
    for i in range(H2):
        for j in range(W2):
            if imos[i][j]!=0:
                break
        else:
            iswhiterow[i]=True
            rowcnt+=1
    cnt-=M*2*rowcnt
    cnt+=W*rowcnt

# 両方に長ければD領域(というか全白かどうか)の確認
if H>N*2 and W>M*2:
    if allwhite:
        cnt=H*W

print(H*W-cnt)
