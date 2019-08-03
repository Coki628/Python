# -*- coding: utf-8 -*-

"""
・最大長方形
・累積和とスタック
"""

import sys
from collections import deque

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

H,W=MAP()
grid=[None]*H
for i in range(H):
    grid[i]=LIST()

# その行を底辺として、高さをどこまで伸ばせるか記録しておく
acc=list2d(H, W, 0)
for j in range(W):
    if grid[0][j]!=1:
        acc[0][j]=1
for j in range(W):
    for i in range(1, H):
        if grid[i][j]!=1:
            acc[i][j]=acc[i-1][j]+1
# 最後にスタックに残ってる分を確認するために番兵的に0を足す
for i in range(H):
    acc[i].append(0)

mx=0
# 各行を高さのヒストグラムとみなす
for i, hist in enumerate(acc):
    stack=deque()
    for j, h in enumerate(hist):
        if not len(stack):
            stack.append((j, h))
        else:
            if stack[-1][1]<h:
                stack.append((j, h))
            elif stack[-1][1]>h:
                # 次のhがスタックより低かったら、そこより手前の最大を確定する
                while len(stack) and stack[-1][1]>h:
                    prev=stack.pop()
                    w2=j-prev[0]
                    h2=prev[1]
                    mx=max(mx, h2*w2)
                stack.append((prev[0], h))
print(mx)
