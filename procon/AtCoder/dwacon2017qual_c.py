# -*- coding: utf-8 -*-

"""
・400点自力AC
"""

import sys
from collections import Counter

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
A=[INT() for i in range(N)]

# なるべく4人ずつ収まるように適切に割り振っていく
C=Counter(A)
cnt=0
cnt+=C[4]
C[4]=0
cnt+=C[2]//2
C[2]%=2
cnt+=min(C[3], C[1])
if C[1]>C[3]:
    C[1]=C[1]-C[3]
    C[3]=0
    if C[2]==1:
        C[2]=0
        C[1]-=2
        cnt+=1
    if C[1]>0:
        cnt+=ceil(C[1], 4)
elif C[3]>C[1]:
    C[3]=C[3]-C[1]
    C[1]=0
    if C[2]==1:
        cnt+=1
    cnt+=C[3]
else:
    if C[2]==1:
        cnt+=1
print(cnt)
