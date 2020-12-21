# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/kyopro_friends/status/1158010219407892480
・候補を総和の約数に絞るのは悪くなかったみたいだ。
・その先、K回以内の操作でいけるかの判定は自力で探せなかったんだが。
・これはKが大きいのに1つずつ数えてるからTLE
"""

import sys
from math import sqrt
from collections import deque

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

N,K=MAP()
A=LIST()

def divisor_set(N: int) -> set:
    """ 約数の列挙・個数 """
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て
    for i in range(2, int(sqrt(N))+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

sm=sum(A)
divs=sorted(divisor_set(sm), reverse=True)

for div in divs:
    B=[]
    for i in range(N):
        if A[i]%div!=0:
            B.append(A[i]%div)
    B.sort()
    B=deque(B)
    cnt=0
    while len(B):
        B[0]-=1
        B[-1]+=1
        if B[0]%div==0:
            B.popleft()
        if B[-1]%div==0:
            B.pop()
        cnt+=1
    if cnt<=K:
        print(div)
        exit()
