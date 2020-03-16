# -*- coding: utf-8 -*-

"""
・自力AC
・丁寧にシミュレーション。同じ要素はまとめて管理。
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

Q, L = MAP()
stack = []
sm = 0
for _ in range(Q):
    q = list(input().split())
    if q[0] == 'Push':
        _, a, b = q
        a, b = int(a), int(b)
        stack.append([a, b])
        sm += a
        if sm > L:
            print('FULL')
            exit()
    elif q[0] == 'Pop':
        _, a = q
        a = int(a)
        sm -= a
        if sm < 0:
            print('EMPTY')
            exit()
        while a > 0 and a >= stack[-1][0]:
            a -= stack[-1][0]
            stack.pop()
        if a > 0:
            stack[-1][0] -= a
            if stack[-1][0] == 0:
                stack.pop()
    elif q[0] == 'Top':
        if not stack:
            print('EMPTY')
            exit()
        print(stack[-1][1])
    else:
        print(sm)
print('SAFE')
