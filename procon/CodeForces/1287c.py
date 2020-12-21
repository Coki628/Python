# -*- coding: utf-8 -*-

"""
・自力WA
・制約に貪欲やめとけって書いてあったのに、下手な貪欲しかできずに終了。。
"""

import sys
from collections import Counter, deque

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

A = [a%2 if a != 0 else -1 for a in A]
C = Counter(A)

total = C[-1]
odd = ceil(N, 2) - C[1]
even = N // 2 - C[0]

def count(A):
    res = 0
    for i in range(N-1):
        if A[i] != A[i+1]:
            res += 1
    return res

ans = INF
B = deque([1] * odd + [0] * even)
for i in range(total):
    A2 = A[:]
    j = k = 0
    while k < total:
        if A2[j] == -1:
            A2[j] = B[k]
            k += 1
        j += 1
    ans = min(ans, count(A2))
    B.append(B.popleft())
B = deque([1] * odd + [0] * even)
for i in range(total):
    A2 = A[:]
    j = k = 0
    while k < total:
        if A2[j] == -1:
            A2[j] = B[k]
            k += 1
        j += 1
    ans = min(ans, count(A2))
    B.appendleft(B.pop())
B = deque([0] * even + [1] * odd)
for i in range(total):
    A2 = A[:]
    j = k = 0
    while k < total:
        if A2[j] == -1:
            A2[j] = B[k]
            k += 1
        j += 1
    ans = min(ans, count(A2))
    B.append(B.popleft())
B = deque([0] * even + [1] * odd)
for i in range(total):
    A2 = A[:]
    j = k = 0
    while k < total:
        if A2[j] == -1:
            A2[j] = B[k]
            k += 1
        j += 1
    ans = min(ans, count(A2))
    B.appendleft(B.pop())
print(ans)
