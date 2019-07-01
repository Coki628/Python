# -*- coding: utf-8 -*-

"""
・二分探索
・真ん中1回切るのが最適かと思ったけど、数が大きくなるとその限りじゃなさそうだ。
・2の10億乗とか絶対そのままやらないんだろうな。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
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


def calc(p, a):
    if p%2 == 0:
        mx = p//2 * (p//2-1)
    else:
        mx = (p//2) ** 2
    return mx >= a

T=INT()

for _ in range(T):
    l,a=MAP()
    low=-1
    hi=l+1
    while low+1<hi:
        mid=(hi+low)//2
        if calc(mid, 2**a):
            hi=mid
        else:
            low=mid
    if hi>l:
        print(0)
    else:
        print(l-hi+1)
