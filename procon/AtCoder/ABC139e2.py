# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=UWbGRhF3Ozw
・解説動画の方針1：変化のあった場所だけ見るようにして高速化
・TLE取れず。
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

N = INT()
A = [deque()]
for i in range(N):
    A.append(deque(LIST()))

cnt = N*(N-1)//2
nxtli = list(range(1, N+1))
day = 0
while cnt > 0:
    prev = cnt
    skip = [False] * (N+1)
    # 前回変更があった場所のみでループを回す
    curli = nxtli[:]
    nxtli = []
    for i in curli:
        if len(A[i]) and not skip[i]:
            a = A[i][0]
            if len(A[a]) and not skip[a]:
                b = A[a][0]
                if i == b:
                    A[i].popleft()
                    A[a].popleft()
                    # 変更があった場所をリストに入れておく
                    nxtli += [i, a]
                    cnt -= 1
                    skip[i] = skip[a] = True
    if prev == cnt:
        print(-1)
        exit()
    day += 1
print(day)
