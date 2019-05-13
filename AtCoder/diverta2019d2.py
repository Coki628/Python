# -*- coding: utf-8 -*-

"""
・速攻ではないけど500点自力AC
・大きい所と小さい所で探し方を分ける系
・参戦中はCで時間くってたから見れなかった。残念。
"""

import sys
from math import sqrt

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

# for i in range(N, 0, -1):
#     print(i, N//i, N%i)

# 0除算対策
if N==1:
    print(0)
    exit()

m=N
ans=set()
while m>=sqrt(N):
    i=N//m
    j=N%m
    if (i-j)%i==0 and N//m==N//(m-1):
        ans.add(m-1)
    m=N//(i+1)

for m in range(1, int(sqrt(N))+1):
    if N//m==N%m:
        ans.add(m)

print(sum(ans))
