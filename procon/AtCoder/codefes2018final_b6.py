# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-festival-2018-final/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2018/11/18/184215
　　　https://atcoder.jp/contests/code-festival-2018-final/submissions/3622951
　　　https://atcoder.jp/contests/code-festival-2018-final/submissions/3627664
・そもそも問題理解できてなかった。
・M配列に書いてある条件を全て満たすような確率ってことか…。
・公式解の分子部分を式変形すると、最終的に N!/(r1!*r2!*..rM!) になる。
　それを元々の分母と合わせて、N!/(r1!*r2!*..rM!*M^N) とすれば解きやすい。
・でかすぎる数を扱うために対数を使う。
・logの世界では掛け算は足し算に、割り算は引き算になる。(そういえばそうだった)
・logの世界で階乗の事前テーブルを作っておく版
・M^Nもlogの世界だとN乗じゃなくて*Nにできたからそれも適用
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians, log10
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
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
def prod(nums): return reduce(mul, nums, 1)
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
rM = LIST()

# 毎回階乗の計算をしなくていいように、logの世界で階乗の事前テーブルを作る
fact_log = [0] * (N+1)
for i in range(1, N+1):
    fact_log[i] = fact_log[i-1] + log10(i)

# 分子
nume = fact_log[N]
# 分母
# M^N は logの世界ではlogM*Nで表せる
deno = log10(M) * N
for i in range(M):
    # logの世界なので掛け算は足し算に
    deno += fact_log[rM[i]]
# 同じく割り算は引き算に
ans = nume - deno
# 10^-x >= p → -x >= ans → x <= -ans を満たすような最大のxを出力
print(ceil(-ans))
