# -*- coding: utf-8 -*-

"""
・D自力AC！
・DPっぽいと思わせておいての貪欲で行ける系だった。
・と思ったら貪欲は嘘解法だった…。。
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

N,K=MAP()
WP=[]
for i in range(N):
    w,p=MAP()
    s=w*p/100
    WP.append((w-s, s))

ans=[0, 0]
taken=[False]*N
for i in range(K):
    w1,s1=ans
    mx=-INF
    nxt=[0, 0]
    idx=-1
    for j in range(N):
        if not taken[j]:
            w2,s2=WP[j]
            if mx<(s1+s2)/(w1+s1+w2+s2):
                nxt[0]=w2
                nxt[1]=s2
                mx=(s1+s2)/(w1+s1+w2+s2)
                idx=j
    taken[idx]=True
    ans[0]+=nxt[0]
    ans[1]+=nxt[1]
print(ans[1]/(ans[0]+ans[1])*100)
