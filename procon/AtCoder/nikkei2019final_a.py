# -*- coding: utf-8 -*-

"""
・累積和
・3000^2=900万のループがきつくて、pythonTLEでpypyACの約0.3秒。
"""

import sys
from itertools import accumulate

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

N=INT()
A=LIST()

acc=[0]+list(accumulate(A))

for i in range(N):
    mx=0
    for j in range(N):
        if j+i>=N:
            break
        mx=max(mx, acc[j+i+1]-acc[j])
    print(mx)
