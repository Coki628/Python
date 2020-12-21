# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/tenka1-2019/editorial.pdf
　　　https://www.youtube.com/watch?v=uQvc7k2xamY
・累積和
・結局、左が黒で右が白っていう形で、どこを境目にするかってだけだった。
・この300点いけなかったのは深く反省…。
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

N=INT()
S=input()

sm1=[0]*(N+2)
for i in range(N):
    sm1[i+1]=sm1[i]
    if S[i]=='#':
        sm1[i+1]+=1
sm2=[0]*(N+2)
for i in range(N+1, 1, -1):
    sm2[i-1]=sm2[i]
    if S[i-2]=='.':
        sm2[i-1]+=1

ans=INF
for i in range(N+1):
    ans=min(ans, sm1[i]+sm2[i+1])
print(ans)
