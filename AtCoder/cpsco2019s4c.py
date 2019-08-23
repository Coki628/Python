# -*- coding: utf-8 -*-

"""
・TLE、まあ10万に普通に階乗絡めてるからそうなるだろうとは思ったが。。
"""

import sys
from bisect import bisect_right
from math import factorial

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(li): return reduce(lcm, li, 1)
def gcd_list(li): return reduce(gcd, li, 0)
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

def nCr(n, r):
    """ 事前テーブルなし組み合わせ簡易版 """
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

N, D = MAP()
A = LIST()

A.sort()
prev = 0
ans = 0
for i in range(N):
    idx = bisect_right(A, A[i]+D)
    n = idx - i
    m = prev - 1
    # 今回分 - 前回との重複分
    ans += nCr(n, 3) - nCr(m, 3)
    prev = n
print(ans)
