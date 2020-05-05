# -*- coding: utf-8 -*-

"""
・自力AC
・切り上げの式変形
・例えば ceil(x, 12) = 9 だったら、区間: (12*8,12*9] がxの有効な範囲になる。
　後は条件の式3つについてそれぞれやって、範囲を狭めればOK。
・ちゃんと考えれば解けるけど、こういう算数ちょこっとみたいなやつ、
　数学強い人は秒で解くからなー。。どうしたもんかねー。
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

A = LIST()
B = LIST()
N = 3

for i in range(N-1):
    B[i+1] += B[i]

ansl = -INF
ansr = INF
for i in range(N):
    l = A[i] * (B[i]-1)
    r = A[i] * B[i]
    ansl = max(ansl, l)
    ansr = min(ansr, r)

if ansl >= ansr:
    print(-1)
else:
    print(ansl+1, ansr)
