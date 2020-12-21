# -*- coding: utf-8 -*-

"""
参考：みんなのツイート
・後ろから累積和、zero sum ranges系
・出なかった。分かれば、すごく最近のABC158Eの類題だった。
・2019を素因数分解して、3と673だったから、3の倍数かつ673の倍数になる場所を探したいなー、
　みたいな、全然違う方向に走ってしまったのが間違いだった。。
"""

import sys
from collections import Counter

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

S = input()
N = len(S)
mod = 2019

acc = [0] * (N+1)
mul = 1
for i in range(N-1, -1, -1):
    x = int(S[i]) * mul
    acc[i] = (acc[i+1] + x) % mod
    mul *= 10
    mul %= mod

C = Counter(acc)
ans = 0
for v in C.values():
    ans += v*(v-1) // 2
print(ans)
