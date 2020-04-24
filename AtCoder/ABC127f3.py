# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc127/editorial.pdf
・WA
・heapqはソートしてくれるけど、真ん中らへんの要素を狙って取るのはできないぽい。
"""

import sys
from heapq import heappush

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

Q=INT()

li=[]
A=B=fx=cost=cnt=l=r=0
for _ in range(Q):
    instr=input()
    if instr[0]=='1':
        _,a,b=map(int, instr.split())
        B+=b
        heappush(li, a)
        if cnt==0:
            l=r=A=a
        else:
            if not (l<=a<=r):
                cost+=min(abs(l-a), abs(r-a))
            # 偶数回目
            if cnt%2==1:
                A=l=li[len(li)//2-1]
                r=li[len(li)//2]
            # 奇数回目
            else:
                A=l=r=li[len(li)//2]
        fx=cost+B
        cnt+=1
    else:
        print(A, fx)
