# -*- coding: utf-8 -*-

import sys

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

for _ in range(INT()):
    N = INT()
    A = LIST()
    if N < 10:
        print(0, 0, 0)
        continue
    g = s = b = 0
    g += 1
    for i in range(1, N):
        if A[i-1] == A[i]:
            g += 1
        else:
            idx = i
            break
    else:
        print(0, 0, 0)
        continue
    s += 1
    for i in range(idx+1, N):
        if A[i-1] == A[i] or s <= g:
            s += 1
        else:
            idx = i
            break
    else:
        print(0, 0, 0)
        continue
    half = N // 2
    if g + s >= half:
        print(0, 0, 0)
        continue
    b += half - idx
    for i in range(half-1, -1, -1):
        if A[i] == A[i+1]:
            b -= 1
        else:
            break
    if b <= 0 or b <= g:
        print(0, 0, 0)
        continue
    print(g, s, b)
