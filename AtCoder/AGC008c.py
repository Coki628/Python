# -*- coding: utf-8 -*-

"""
・自力WA。。
"""

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

def check1(I, J, L):
    res = 0
    res += I - I%2
    res += J - J%2
    res += L - L%2
    if I%2 and J%2 and L%2:
        res += 3
    return res

def check2(I, J, L):
    cnt = min(I, J, L)
    I -= cnt
    J -= cnt
    L -= cnt
    res = cnt * 3
    res += I - I%2
    res += J - J%2
    res += L - L%2
    return res

I, O, T, J, L, S, Z = MAP()

ans = O
ans += max(check1(I, J, L), check2(I, J, L))
print(ans)
