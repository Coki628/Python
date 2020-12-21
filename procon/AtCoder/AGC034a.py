# -*- coding: utf-8 -*-

"""
・400点自力AC
・方針は最初から合ってたけど、際の値の微調整で2WAしたのはちょっともったいない。
"""

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

N,A,B,C,D=MAP()
S='#'+input()+'#'

if C<D:
    for i in range(A, D-1):
        if S[i]==S[i+1]=='#':
            No()
            exit()
    Yes()
else:
    for i in range(A, C-1):
        if S[i]==S[i+1]=='#':
            No()
            exit()
    for i in range(B, D+1):
        if S[i-1]==S[i]==S[i+1]=='.':
            Yes()
            exit()
    No()
