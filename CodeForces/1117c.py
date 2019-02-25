# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/65365
・前計算と二分探索
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product, accumulate
from operator import itemgetter, mul
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
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

x,y=MAP()
init=(x,y)
x,y=MAP()
dest=(x,y)
N=INT()
S=input()

# まず、経過時間によって風の影響をどの程度受けるかを出しておく
d=[None]*(N+1)
d[0]=(0,0)
for i in range(N):
    x,y=d[i]
    if S[i]=='L':
        x-=1
    elif S[i]=='R':
        x+=1
    elif S[i]=='U':
        y+=1
    else:
        y-=1
    d[i+1]=(x,y)

def calc(k):
    # 商と余りをうまく使って循環する中から経過時間に対応する距離を割り出す
    a=k//N
    b=k%N
    x=init[0]+d[N][0]*a+d[b][0]
    y=init[1]+d[N][1]*a+d[b][1]
    dist=abs(x-dest[0])+abs(y-dest[1])
    return dist

low=0
# 範囲は十分に大きい数にしておく(10**18でも大丈夫だった)
hi=10**30
# low+1 == hi -> lowとhiが隣り合った状態なので、境界が確定する
while low+1<hi:
    mid=(hi+low)//2
    if calc(mid)<=mid:
        hi=mid
    else:
        low=mid
# いくら進んでも辿り着けない
if hi==10**30:
    print(-1)
else:
    # dist<=midを初めて満たす(初めてdistがmid以上になる)位置
    print(hi)

# 以下(境界の下)を探す時：f(mid)<mid, ->low
# ex.) abc020c, kupc2018d(ちょっと変形)
# 以上(境界の上)を探す時：f(mid)<=mid, ->hi
# ex.) abc063d, cf1117c, alds1_4_d
# 未満(境界の下)を探す時：f(mid)<=mid, ->low
