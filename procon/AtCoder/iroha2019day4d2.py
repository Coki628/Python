# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day4/editorial-D.pdf
・二分探索、使って損しない数の考察、logの世界での比較
"""

import sys
from math import log2

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


def calc(m, a):
    if m<=0:
        return False
    if m==1:
        return a==0
    mx=0
    # 2,5,3を使う数ループ
    for i in range(2):
        for j in range(2):
            for k in range(5):
                # 敵の強さ
                p=pow(2, i)*pow(5, j)*pow(3, k)
                # 使うマス目の数
                q=2*i+5*j+3*k+i+j+k-1
                # 使うマス目が多すぎ
                if q>m:
                    continue
                # 4のために使える、残りのマス目の数
                rest=m-q
                c=rest//5
                # p*4**cのlog
                mx=max(mx, log2(p)+2*c)
    # p*4**c>=2**aの比較をlogの世界でやる
    return mx>=a

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
        if calc(mid, a):
            hi=mid
        else:
            low=mid
    if hi>l:
        print(0)
    else:
        print(l-hi+1)
