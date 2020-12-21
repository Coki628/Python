# -*- coding: utf-8 -*-

"""
・300点自力AC。結構手こずって危なかった。
・ビット全探索、なのはすぐ思ったけど、中の条件判定が意外と素直じゃない感じ。
・正直なはずの誰かが正直だって言った人は、正しいことを言ってないといけないので、
　それを全部探しに行けるように再帰探索させた。
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

def bit_count(i):

    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

N = INT()
A = []
for i in range(N):
    a = []
    for j in range(INT()):
        x, y = MAP()
        a.append((x-1, y))
    A.append(a)

def check(a):
    for x, y in a:
        if honest[x] == -1:
            honest[x] = y
            if y == 1:
                if not check(A[x]):
                    return False
        elif honest[x] != y:
            return False
    return True

ans = 0
for S in range(1<<N):
    honest = [-1] * N
    valid = True
    for i in range(N):
        if S & (1<<i):
            honest[i] = 1
    for i in range(N):
        if S & (1<<i):
            if not check(A[i]):
                valid = False
    if valid:
        ans = max(ans, bit_count(S))
print(ans)
