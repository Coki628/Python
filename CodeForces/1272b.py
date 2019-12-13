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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    S = input()
    C = Counter(S)
    if (C['L'] == 0 or C['R'] == 0) and (C['U'] == 0 or C['D'] == 0):
        print(0)
    elif (C['L'] == 0 or C['R'] == 0):
        print(2)
        print('UD')
    elif (C['U'] == 0 or C['D'] == 0):
        print(2)
        print('LR')
    else:
        lr = min(C['L'], C['R'])
        ud = min(C['U'], C['D'])
        print(lr * 2 + ud * 2)
        ans = ''
        ans += 'L' * lr
        ans += 'U' * ud
        ans += 'R' * lr
        ans += 'D' * ud
        print(ans)
