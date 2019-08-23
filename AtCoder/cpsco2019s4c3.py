# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/cpsco2019-s4/editorial.pdf
・公式解
・1人目を固定してあと2人の組み合わせにすればいいって話。
"""

import sys
from bisect import bisect_right

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

N, D = MAP()
A = LIST()

A.sort()
prev = 0
ans = 0
for i in range(N):
    idx = bisect_right(A, A[i]+D)
    n = idx - i
    ans += (n-1)*(n-2)//2
print(ans)
