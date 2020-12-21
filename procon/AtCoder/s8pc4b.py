# -*- coding: utf-8 -*-

"""
・累積MAX
・公式解ではない。条件色々整えるの大変だったのでお勧めしない感じのやり方。
"""

import sys
from itertools import combinations, accumulate

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
A=LIST()

# 累積MAX
acc=list(accumulate(A, max))

ans=INF
for comb in combinations(range(N), K):
    sm=0
    for i, n in enumerate(comb):
        if i==0:
            # 1つめは累積MAXとの差分
            cost=max(acc[n]-A[n], 0)
            # 一番左じゃなくて、1つ手前の累積MAXより高くないなら+1
            if n!=0 and acc[n-1]==acc[n]:
                cost+=1
        else:
            # 2つめ以降は、max(1つ前に選んだやつ, 1つ手前の累積MAX)との差分+1でmax取る
            cost=max(max(prev, acc[n-1])-A[n]+1, 0)
        sm+=cost
        prev=A[n]+cost
    ans=min(ans, sm)
print(ans)
