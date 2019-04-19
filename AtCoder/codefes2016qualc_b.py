# -*- coding: utf-8 -*-

"""
・優先度付きキュー
・想定解とは違うけどちゃんとACではある。
・200点なのに実装時間かかったし、良い手ではないとは思うが。
"""

import sys
from heapq import heappop, heappush, heapify

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

K,T=MAP()
A=LIST()
A=[(-A[i], i) for i in range(T)]
# heapqで残数が多い順に並べておく
heapify(A)

prev=-1
while len(A):
    v,i=heappop(A)
    # 前の同じ種類でなければそれを食べる
    if i!=prev:
        v+=1
        prev=i
        if v!=0:
            heappush(A, (v, i))
    # 同じ種類だったら、もう1つ出してきてそっちを食べる
    else:
        # 他のを出そうとしてもう何もなければNGとして終了
        if len(A)==0:
            print(-v)
            exit()
        v2,i2=heappop(A)
        v2+=1
        prev=i2
        if v2!=0:
            heappush(A, (v2, i2))
        # 食べなかった方も戻す
        heappush(A, (v, i))
# 最後までなくなればOKとして終了
print(0)
