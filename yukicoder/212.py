"""
・自力AC
・確率・期待値系
・まず1回分の期待値を出す。で、今回は各サイコロの目の積が欲しいので、1回期待値^回数の形になる。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

p, c = MAP()

A = [2, 3, 5, 7, 11, 13]
B = [4, 6, 8, 9, 10, 12]

ans1 = 0
for a in A:
    ans1 += a / 6
ans2 = 0
for b in B:
    ans2 += b / 6

ans = ans1**p * ans2**c
print(ans)
