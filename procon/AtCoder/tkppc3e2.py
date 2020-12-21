# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/tkppc3/editorial.pdf
　　　https://mathtrain.jp/expectation
　　　https://atarimae.biz/archives/14396
　　　https://kimiyuki.net/writeup/algo/atcoder/tkppc3-e/
・期待値系
・期待値の線形性
・以下、色々噛み砕く。
・1面の期待値は(N+1)/2 => 例えばN=5なら1～5の真ん中の3
・それに出た面数をかければ和の期待値
・ある面数kで終わる確率は、k回違う値が出て、k+1回目でダブるってことなので
　n/n(最初は何が出てもいい)*n-1/n(前に出たやつ以外)*...
　*n-(k-1)/n(k回目はそれまでに出たk-1種類以外)
　*k/n(k+1回目はそれまでに出たk種類のどれか、ここだけ計算が違う)
　これで出せる。
・和の期待値 * その面数で終わる確率 を各回分足し合わせる。
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

N=INT()

# まずは1面の期待値
e1=(N+1)/2

memo=[0]*(N+1)
memo[0]=1
# 1～k回目の確率は規則的なのでメモに入れておく
for i in range(N):
    memo[i+1]=memo[i]*((N-i)/N)

ans=0
# 1～N面で終わる各ケースについて足して合わせていく
for k in range(1, N+1):
    # 1面の期待値*面数*1～k回目の確率*k+1回目の確率
    ans+=e1*k*memo[k]*k/N
print(ans)
