# -*- coding: utf-8 -*-

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

N,M=MAP()

ans=0
if N==M:
    ans=0
elif N>M:
    if N%M==0:
        ans=0
    else:
        if N%2==0 and M%2==0:
            ans=M-2
        else:
            ans=M-1
elif N<M:
    if M%N==0:
        ans=N
    else:
        if N%2==0 and M%2==0:
            ans=M-2
        else:
            ans=M-1
print(ans)
