# -*- coding: utf-8 -*-

"""
参考：https://qiita.com/drken/items/03c7db44ccd27820ea0d
・最長共通部分列問題
・i側とj側の両方向に配るDP
・復元
・計算量O(len(s)*len(t)+len(s)+len(t))=900万ちょっと、pypyで0.6秒くらい。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def ceil(a, b=1): return int(-(-a // b))
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

s = input()
t = input()

dp = [[0] * (len(t)+1) for i in range(len(s)+1)]

for i in range(len(s)):
    for j in range(len(t)):
        # 下から右に送る
        dp[i+1][j+1] = max(dp[i+1][j+1], dp[i+1][j])
        # 右から下に送る
        dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j+1])
        # 今回増やす文字が一致するなら、LCSが1文字増える
        if s[i] == t[j]:
            # 左上から右下に送る
            dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j]+1)

# 復元(DPテーブルを遡る)
i = len(s)
j = len(t)
ans = ''
while i > 0 and j > 0:
    # 上が同じ値なら上に移動
    if dp[i-1][j] == dp[i][j]:
        i -= 1
    # 左が同じ値なら左に移動
    elif dp[i][j-1] == dp[i][j]:
        j -= 1
    # 上も左も値が異なれば、そこが更新箇所なので、該当位置の文字を取得
    else:
        ans = s[i-1] + ans
        i -= 1
        j -= 1
print(ans)
