# -*- coding: utf-8 -*-

"""
・600点自力AC
・とはいえこれ1000人以上通してたから何とも…。
・終了間際で通せたのは良いけど、もっと早くやれても良かった。
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

S=list(input())
N=len(S)

cnt=cnta=0
i=0
while i<N:
    # 直前に連続するAをカウントしておく
    if S[i]=='A':
        cnta+=1
    else:
        cnta=0
    if i+2<N:
        if S[i]=='A' and S[i+1]=='B' and S[i+2]=='C':
            # Aが連続していたほどBCがたくさん左に動く
            cnt+=cnta
            S[i]='B'
            S[i+1]='C'
            S[i+2]='A'
            i+=2
            cnta-=1
            continue
    i+=1
print(cnt)
