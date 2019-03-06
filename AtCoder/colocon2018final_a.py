# -*- coding: utf-8 -*-

"""
・300点自力AC
・絶対もっとシンプルにやる方法がある気がした。
　まあ通ったからOK。
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
from datetime import date

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

N=INT()
S=input()

# 1からnまでの等差数列の和
def get_sum(n):
    return (1+n)*n//2

# Bが1つもなかったらずっと増え続ける
if not S.count('B'):
    print(get_sum(len(S)*N))
    exit()

# 境界にあるAが計算しずらいのでひとまず前に移動させておく
S=deque(S)
cnt1=0
while True:
    s=S.pop()
    if s=='B':
        S.append(s)
        break
    S.appendleft(s)
    cnt1+=1
S=''.join(S)
# 移動させた個数(cnt1)と、それを含む場所のAの総数(idx)を取っておく
idx=S.find('B')

cnt2=0
C=Counter()
# 連続した個数毎にまとめる
for i in range(len(S)):
    if S[i]=='A':
        cnt2+=1
    else:
        if cnt2!=0:
            C[cnt2]+=1
        cnt2=0

ans=0
# 連続した個数毎に和を計算
for k, v in C.items():
    ans+=get_sum(k)*v*N
# 移動させた部分の帳尻を合わせる
ans-=get_sum(idx)
ans+=get_sum(cnt1)
ans+=get_sum(idx-cnt1)

print(ans)
