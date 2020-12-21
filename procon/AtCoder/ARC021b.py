# -*- coding: utf-8 -*-

"""
・自力AC！
・XOR系
・a^a=0 とか a^b=c => a=b^c とかをうまく使って考える。
"""

import sys
from itertools import accumulate
from operator import xor

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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
B = [INT() for i in range(N)]

# 全部xorして0にならなければ矛盾しているのでNG
acc = list(accumulate(B, xor))
if acc[-1] != 0:
    print(-1)
    exit()

# 辞書順最小に従って、A[0] = 0 を決め打ちすれば、あとはそこから全部導ける
A = [0] * N
for i in range(N-1):
    A[i+1] = A[i] ^ B[i]

[print(a) for a in A]
