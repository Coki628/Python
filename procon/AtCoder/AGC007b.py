# -*- coding: utf-8 -*-

"""
・なんとか自力AC。考察時間かかったけど。。400かぁ。
・最初、A,B内での制約を守りつつ各要素の間隔を十分に離しておく。
　この時点でA+Bは同値になるようにする。
・あとはA+Bの制約に従うように少しずつ要素に加算をしていけばOK。
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

N = INT()
P = [p-1 for p in LIST()]

A = [0] * N
B = [0] * N
# 最大要素が制約(10^9)を超えない範囲で各要素の間隔を十分(N<=2万なので2万以上)に広げる
gap = 30000
for i in range(N):
    # この時点でA+Bはどの要素も等しくしておく
    A[i] = gap * i + 1
    B[i] = gap * (N-i)

# 和を小さくしたい方から順に+iしていく(各要素が十分離れているので、数列A,B内での制約は崩れない)
for i, p in enumerate(P):
    A[p] += i

print(*A)
print(*B)
