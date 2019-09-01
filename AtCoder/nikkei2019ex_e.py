# -*- coding: utf-8 -*-

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()

for i in range(1, N+1):
    ans = ''
    if i%2 == 0:
        ans += 'a'
    if i%3 == 0:
        ans += 'b'
    if i%4 == 0:
        ans += 'c'
    if i%5 == 0:
        ans += 'd'
    if i%6 == 0:
        ans += 'e'
    if ans == '':
        ans = str(i)
    print(ans)
