# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc039
・あまりに冗長だなぁと思ったので、解説を確認。
・とりあえず、AとB逆位置がないので、半分は余計だった。。
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

A,B=input().split()

mx=int(A)-int(B)

A2='9'+A[1]+A[2]
mx=max(mx, int(A2)-int(B))
A2=A[0]+'9'+A[2]
mx=max(mx, int(A2)-int(B))
A2=A[0]+A[1]+'9'
mx=max(mx, int(A2)-int(B))

B2='1'+B[1]+B[2]
mx=max(mx, int(A)-int(B2))
B2=B[0]+'0'+B[2]
mx=max(mx, int(A)-int(B2))
B2=B[0]+B[1]+'0'
mx=max(mx, int(A)-int(B2))

print(mx)
