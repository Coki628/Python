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

S=input()
N=len(S)

cnt2=cnt5=ans=0
for i in range(N):
    if S[i]=='2':
        cnt2+=1
    else:
        cnt5+=1
    if cnt5>cnt2:
        print(-1)
        exit()
    ans=max(ans, cnt2-cnt5)
if cnt2!=cnt5:
    print(-1)
else:
    print(ans)
