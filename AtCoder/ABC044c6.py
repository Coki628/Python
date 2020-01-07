# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・N<=50に半分全列挙で挑むもTLE。。(C++でもTLE)
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

N, K = MAP()
# 総和0を平均KとするためKを引く
A = [a-K for a in LIST()]

# 半分全列挙
A1 = A[:N//2]
A2 = A[N//2:]

# それぞれの全組み合わせ
P1 = [0]
for S in range(1, 1<<len(A1)):
    p = 0
    for i in range(len(A1)):
        if S & 1<<i:
            p += A1[i]
    P1.append(p)
P2 = [0]
for S in range(1, 1<<len(A2)):
    p = 0
    for i in range(len(A2)):
        if S & 1<<i:
            p += A2[i]
    P2.append(p)

# 総和0になるペアの数を求める
C2 = Counter(P2)
ans = 0
for p in P1:
    ans += C2[-p]
# どちらも1つも選ばない分の1を引く
print(ans - 1)
