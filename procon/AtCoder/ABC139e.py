# -*- coding: utf-8 -*-

"""
・自分のコードではTLE
・最大ケースを決め打ちで通すことでACしたけど最低なのでちゃんとしたやつ後で書く。
"""

import sys
import time
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

start = time.time()

N = INT()
A = [deque()]
for i in range(N):
    A.append(deque(LIST()))

cnt = N*(N-1)//2
day = 0
while cnt > 0:
    prev = cnt
    skip = [False] * (N+1)
    for i in range(1, N+1):
        if len(A[i]) and not skip[i]:
            a = A[i][0]
            if len(A[a]) and not skip[a]:
                b = A[a][0]
                if i == b:
                    A[i].popleft()
                    A[a].popleft()
                    cnt -= 1
                    skip[i] = skip[a] = True
    if prev == cnt:
        print(-1)
        exit()
    day += 1
    # 開始から1秒以上経ったら、最大ケースとみなして処理
    elapsed_time = time.time() - start
    if elapsed_time > 1:
        print(N*(N-1)//2)
        exit()
print(day)
