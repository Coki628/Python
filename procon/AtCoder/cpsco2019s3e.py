# -*- coding: utf-8 -*-

"""
・自力AC!
・XOR系、ビット毎に管理
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
A = LIST()
MAX = 30

bits = [0] * (MAX+1)
X = 0
for i, a in enumerate(A):
    # Xと和の状態を更新
    X ^= a
    for k in range(MAX+1):
        if (a>>k) & 1:
            bits[k] += 1
    ans = 0
    for k in range(MAX+1):
        cnt1 = cnt2 = 0
        if (X>>k) & 1:
            # 和の方でこのビットが立っている数だけ、XとのXORで減る
            cnt1 = bits[k]
            # 和の方でこのビットが立っていない数だけ、XとのXORで増える
            cnt2 = (i+1) - bits[k]
        ans += 2**k * (bits[k] - cnt1 + cnt2)
    print(ans)
