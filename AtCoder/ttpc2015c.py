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

S = input()

# 'o'が多いものから置換していく
for cnto in range(100, 2, -1):
    T1 = 'o'*cnto + 'kayama'
    if cnto % 3 == 0:
        T2 = 'Ookayama'
    elif cnto % 3 == 1:
        T2 = 'okayama'
    else:
        T2 = 'Okayama'
    S = S.replace(T1, T2)
print(S)
