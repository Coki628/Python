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

N, Q = MAP()

li = list2d(N, N, 0)
for _ in range(Q):
    q = LIST()
    if q[0] == 1:
        _, a, b = q
        a -= 1; b -= 1
        li[a][b] = 1
    elif q[0] == 2:
        _, a = q
        a -= 1
        for i in range(N):
            if li[i][a]:
                li[a][i] = 1
    else:
        _, a = q
        a -= 1
        S = set()
        for i in range(N):
            if li[a][i]:
                S.add(i)
        # 現時点での情報を集めきってから更新する
        for i in S:
            for j in range(N):
                if li[i][j] and j != a:
                    li[a][j] = 1

ans = list2d(N, N, '')
for i in range(N):
    for j in range(N):
        if li[i][j]:
            ans[i][j] = 'Y'
        else:
            ans[i][j] = 'N'
for i in range(N):
    print(''.join(ans[i]))
