# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/bitflyer2018-final/editorial.pdf
　　　http://drken1215.hatenablog.com/entry/2018/07/01/133700
　　　https://www.hamayanhamayan.com/entry/2018/07/01/093213
・多分各クエリ二分探索だよなーは思ったけど、うまい方法に辿り着けず。
・人で4つの区分に分けるって形にできなかった。
・というか全額支給の人をまとめて計算する方法が出なかった。
・今回は境界値はどっちに入れても大丈夫系だったから、こういう時は
　区間の取り方は統一すると管理しやすい。
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

N,Q=MAP()
X=LIST()

# 累積和で事前準備
sm=[0]+list(accumulate(X))

for _ in range(Q):
	c,d=MAP()
	ans=0
	# i,j,kを全部統一(今回は半開区間"[a,b)")して扱うと二分探索しやすい
	i=bisect_left(X, c-d)
	# 前半の定額
	ans+=d*i
	j=bisect_left(X, c)
	# 前半の全額支給
	# c-Xi,c-Xi+1..と足し上げるので、まとめると(c*人数)-(Xi..Xj-1の和)になる
	ans+=c*(j-i)-(sm[j]-sm[i])
	k=bisect_left(X, c+d)
	# 後半の全額支給
	# Xj-c,Xj+1-c.. => (Xj..Xk-1の和)-(c*人数)
	ans+=(sm[k]-sm[j])-c*(k-j)
	# 後半の定額
	ans+=d*(N-k)
	print(ans)
