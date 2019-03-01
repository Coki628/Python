# -*- coding: utf-8 -*-

"""
参考：https://www.hamayanhamayan.com/entry/2018/02/11/101334
　　　http://algon-320.hatenablog.com/entry/2018/02/11/022639
　　　https://img.atcoder.jp/yahoo-procon2018-qual/editorial.pdf
・bitDP
・ゲーム木の探索(ミニマックス)
・メモ化再帰
・bitDP初めてだったのもあるけど、噛み砕くの大変だったー。
・O(2^N*N^2)=26万くらい*18^2=8500万はpypyでも無理かと…。
・c++で0.6秒くらいでAC
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
X=LIST()
C=LIST()
V=LIST()

# dp[n][msk] := n個売却して商品集合msk全てを購入した場合の最大価値(購入できないなら0)
# メビウス変換を行うことで以下のようになる
# dp[n][msk] := n個売却して購入可能商品がmskの場合の最大価値
dp=list2d(N+1, 1<<N, 0)

for n in range(N+1):
    # n個売却して円を獲得
    yen=0
    for i in range(n):
        yen+=X[i]
    # どの商品を購入するかをビットで全組み合わせ試す
    for msk in range(1<<N):
        cost=val=0
        for i in range(N):
            # 1<<iビット目の商品を購入
            if msk&(1<<i):
                cost+=C[i]
                val+=V[i]
        # 円が足りていればmskの全商品を購入する
        if cost <= yen:
            dp[n][msk]=val
        else:
            dp[n][msk]=0
    
    for msk in range(1<<N):
        for i in range(N):
            if msk&(1<<i):
                # dp[n][msk]の部分集合とmaxを取る(メビウス変換：状態xの部分状態yを全列挙)
                dp[n][msk]=max(dp[n][msk], dp[n][msk^(1<<i)])

memo=list2d(N+1, 1<<N, -1)
def rec(n, turn, msk):
    # メモ返却と終了条件
    if memo[n][msk]!=-1:
        return memo[n][msk]
    if n==N:
        return 0
    # 先手番(売却か購入)
    if turn==0:
        ret=0
        # 売却
        ret=max(ret, rec(n+1, 1, msk))
        # 購入(今の状況で得られる最大値はbitDPで前計算してある)
        ret=max(ret, dp[n][msk])
    # 後手番(破棄)
    else:
        ret=INF
        for i in range(N):
            # 1<<iビット目の商品がまだあったら破棄してみる
            if msk&(1<<i):
                # 破棄(一番最小化できるものを使う)
                ret=min(ret, rec(n, 0, msk-(1<<i)))
    memo[n][msk]=ret
    return ret

print(rec(0, 0, (1<<N)-1))
