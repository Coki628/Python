# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・bitset高速化、ビットで累積和っぽいもの
・多倍長いけないかなーって思ったけど、20万ビットはさすがにTLE。。
　てか多分ビットカウントが間に合わないんだと思う。
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

def bit_count(i):

    return bin(i).count('1')

N, M, Q = MAP()

# 出発前の列車の集合
S1 = [(1<<M)-1] * (N+1)
# 終着後の列車の集合
S2 = [0] * (N+1)
for i in range(M):
    l, r = MAP()
    l -= 1; r -= 1
    S1[l+1] ^= 1<<i
    S2[r] |= 1<<i
for i in range(1, N+1):
    S1[i] &= S1[i-1]
    S2[i] |= S2[i-1]

for i in range(Q):
    p, q = MAP()
    p -= 1; q -= 1
    pq = S1[p] & S2[q]
    print(bit_count(pq))
