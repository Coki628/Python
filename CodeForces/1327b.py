# -*- coding: utf-8 -*-
 
import sys
from collections import Counter
 
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

def check(A):
    for i, a in enumerate(A):
        if a == -1:
            for x in range(1, N+1):
                if x not in taken:
                    ans = (i+1, x)
                    return ans
    return ()
 
for _ in range(INT()):
    N = INT()
    A = [-1] * N
    taken = set()
    for i in range(N):
        li = LIST()
        k, li = li[0], li[1:]
        for a in li:
            if a not in taken:
                A[i] = a
                taken.add(a)
                break
    ans = check(A)
    if ans:
        print('IMPROVE')
        print(*ans)
    else:
        print('OPTIMAL')
