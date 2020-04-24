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

x, N = MAP()
H = [0] * (N-1)
for i in range(N-1):
    H[i] = INT()

H.sort(reverse=1)
place = 1
for h in H:
    if h < x:
        break
    else:
        place += 1

place = str(place)
if place[-1] == '1':
    print('{0}st'.format(place))
elif place[-1] == '2':
    print('{0}nd'.format(place))
elif place[-1] == '3':
    print('{0}rd'.format(place))
else:
    print('{0}th'.format(place))
