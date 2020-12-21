# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc126/editorial.pdf
　　　https://www.youtube.com/watch?v=26AWkQNRb-A
・XOR系
・2べき-1の全部XORは常に0
・同じ値のXORは打ち消し合う。
・こういう構築系は、ガチャガチャ色々作ってみて規則性を探し出すこと。
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

M,K=MAP()

if 2**M<=K:
    print(-1)
    exit()

if M==0:
    print(0, 0)
    exit()

if M==1:
    if K==0:
        print(0, 0, 1, 1)
    else:
        print(-1)
    exit()

ans=[]
for i in range(2**M):
    if i!=K:
        ans.append(i)
ans.append(K)
for i in range(2**M-1, -1, -1):
    if i!=K:
        ans.append(i)
ans.append(K)

print(*ans)
