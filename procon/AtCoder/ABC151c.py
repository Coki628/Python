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

N, M = MAP()
ac = [0] * N
pn = [0] * N
for i in range(M):
    p, s = input().split()
    p = int(p) - 1
    if s == 'AC':
        ac[p] = 1
    elif s == 'WA' and not ac[p]:
        pn[p] += 1
# ACしなかった問題のペナは0にする
for i in range(N):
    if not ac[i]:
        pn[i] = 0

print(sum(ac), sum(pn))
