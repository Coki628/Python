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
INF = float('inf')
MOD = 10 ** 9 + 7

for _ in range(INT()):
    A = list(map(int, list(input())))
    N = len(A)
    odd = []
    even = []
    for a in A:
        if a % 2 == 0:
            even.append(a)
        else:
            odd.append(a)
    odd = odd[::-1]
    even = even[::-1]
    ans = []
    while odd and even:
        if odd[-1] < even[-1]:
            ans.append(odd.pop())
        else:
            ans.append(even.pop())
    while odd:
        ans.append(odd.pop())
    while even:
        ans.append(even.pop())
    ans = list(map(str, ans))
    print(''.join(ans))
