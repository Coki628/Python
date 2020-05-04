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

D = { 'B1': 8,'B2': 7,'B3': 6,'B4': 5,'B5': 4,'B6': 3,'B7': 2,'B8': 1,'B9': 0, \
        '1F': 9,'2F': 10,'3F': 11,'4F': 12,'5F': 13,'6F': 14,'7F': 15,'8F': 16,'9F': 17, }

S, T = input().split()

ans = abs(D[S] - D[T])
print(ans)
