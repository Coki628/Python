# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/09/08/195000
　　　http://kyopro.hateblo.jp/entry/2018/09/10/165511
・二分探索
・ほぼ写経なのであんま満足してない。
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

def calc(x):
    lim=a*b-1
    mx=(x+1)//2*(x+1-(x+1)//2)
    return mx>lim

Q=INT()
for _ in range(Q):
    a,b=MAP()

    # 例外処理
    if a==b:
        print((a-1)*2)
        continue

    # 二分探索
    low=1
    hi=10**18
    while low+1<hi:
        mid=(hi+low)//2
        if calc(mid):
            hi=mid
        else:
            low=mid
    print(low-1)
