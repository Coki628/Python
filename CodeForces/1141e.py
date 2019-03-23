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

H,N=MAP()
D=LIST()

sm=[0]*N
mn=sm[0]=D[0]
for i in range(1, N):
    sm[i]=sm[i-1]+D[i]
    mn=min(mn, sm[i])

if sm[-1]>=0 and mn>H:
    print(-1)
    exit()

# i=0
# while True:
#     h+=mn+sm[-1]*i
#     if h<=0:
#         break
#     i+=1

ans=(H//abs(sm[-1]))*N
print()
