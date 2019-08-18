# -*- coding: utf-8 -*-

"""
・これはTLE
・毎回dictを初期化のためにdeepcopyしてたのが遅かったと思われる。
"""

import sys
from collections import deque, defaultdict
from copy import deepcopy

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

S = input()
T = input()
N = len(S)

d = defaultdict(deque)
for i, s in enumerate(S):
    d[s].append(i)

d2 = deepcopy(d)
cur = -1
ans = 0
for i, t in enumerate(T):
    if t not in d2:
        print(-1)
        exit()
    if len(d2[t]) == 0:
        ans += N
        d2 = deepcopy(d)
        cur = -1 
    idx = d2[t].popleft()
    if idx < cur:
        ans += N
        d2 = deepcopy(d)
    cur = idx
ans += cur+1
print(ans)
