# -*- coding: utf-8 -*-

"""
・500点自力AC！
・とはいえ貪欲に突っ走るだけだった気がするから、これはまあ、ね。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, L = MAP()
A = LIST()

# 結び目のリストを作っておく
B = []
for i in range(1, N):
    B.append(A[i-1]+A[i])
mx = max(B)
# 一番長いところがL未満なら無理
if mx < L:
    print('Impossible')
    exit()

# 一番長いところに集めてくるように切る
print('Possible')
idx = B.index(mx)
for i in range(idx):
    print(i+1)
for i in range(N-2, idx, -1):
    print(i+1)
print(idx+1)
