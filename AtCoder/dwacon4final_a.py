# -*- coding: utf-8 -*-

"""
参考：https://ferin-tech.hatenablog.com/entry/2018/02/07/053155
　　　https://www.hamayanhamayan.com/entry/2018/02/05/015056
・1秒ずつシミュレーション
・単位は全部秒で管理
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

H,M,S=MAP()
C1,C2=MAP()

start=H*3600+M*60+S
now=start
l=INF
r=0
cnt1=cnt2=0
# 1秒ずつ、条件の範囲を超えるまでシミュレーションしていく
while True:
    now+=1
    # 12時ちょうどで全部重なっているときは含まない
    if now%60*60==now%3600 and now%3600*12==now%(3600*12):
        continue
    # 秒針と分針：時計一周を3600としてどの位置にあるかで、秒針が分針を追い抜く瞬間を見つける
    # 現在-1時点の秒針<=現在-1時点の分針 and 現在時点の秒針>現在時点の分針
    if (now-1)%60*60<=(now-1)%3600 and now%60*60>now%3600:
        cnt1+=1
    # 分針と時針：時計一周を3600*12としてどの位置にあるかで、分針が時針を追い抜く瞬間を見つける
    if (now-1)%3600*12<=(now-1)%(3600*12) and now%3600*12>now%(3600*12):
        cnt2+=1
    # 重なった回数が条件を満たす間はここを通る
    if cnt1==C1 and cnt2==C2:
        l=min(l, now-start)
        r=max(r, now-start)
    # 条件の範囲を超えたらそれ以上やる必要ない
    if cnt1>C1 or cnt2>C2:
        break

if l==INF:
    print(-1)
else:
    print(l, r)
