# -*- coding: utf-8 -*-

"""
・蟻本演習2-4-1、自力ならず。。
・優先度付きキュー
・heapq使うって言うヒントがありながら、昔やったやつを通せなかったのは悔しい。
"""

import sys
from collections import deque
from heapq import heapify, heappush, heappop

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

A1 = A[:N]
A2 = A[N:N*2]
A3 = [-a for a in A[N*2:]]
heapify(A1)
A2 = deque(A2)
heapify(A3)

for i in range(N):
    # 左グループの最小を除いて真ん中グループの左端を追加する場合
    res1 = A2[0] - A1[0]
    # 右グループの最大を除いて真ん中グループの右端を追加する場合
    res2 = -A3[0] - A2[-1]
    # 得をする方の処理を行う
    if res1 >= res2:
        heappush(A1, A2.popleft())
        heappop(A1)
    else:
        heappush(A3, -A2.pop())
        heappop(A3)
print(sum(A1)-(-sum(A3)))
