# -*- coding: utf-8 -*-

"""
・自力AC
・dequeでシミュレーション
"""

import sys
from collections import deque

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

que = deque(A)
cur = 0
ans = []
while que:
    if que[0] <= que[-1]:
        if que[0] > cur:
            cur = que.popleft()
            ans.append('L')
            continue
        elif que[-1] > cur:
            cur = que.pop()
            ans.append('R')
            continue
        else:
            break
    else:
        if que[-1] > cur:
            cur = que.pop()
            ans.append('R')
            continue
        elif que[0] > cur:
            cur = que.popleft()
            ans.append('L')
            continue
        else:
            break
print(len(ans))
print(''.join(ans))
