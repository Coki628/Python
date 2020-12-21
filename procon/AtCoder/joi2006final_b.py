# -*- coding: utf-8 -*-

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

N, K = MAP()
A = LIST(K)

# 白紙カードの有無で場合分け
A.sort()
cheat = False
if A[0] == 0:
    cheat = True
    A.pop(0)
    K -= 1

# ランレングス圧縮で、連続した列毎にまとめる
s = A[0]
B = []
mx = 0
for i in range(1, K):
    if A[i-1] + 1 != A[i]:
        t = A[i-1]
        B.append((s, t))
        mx = max(mx, t - s + 1)
        s = A[i]
t = A[-1]
B.append((s, t))
mx = max(mx, t - s + 1)

# 白紙カードありなら、全てのまとまりの間に、それを挟めるか確認する
if cheat:
    M = len(B)
    for i in range(M-1):
        s1, t1 = B[i]
        s2, t2 = B[i+1]
        if t1 + 2 == s2:
            mx = max(mx, t2 - s1 + 1)
print(mx)
