# -*- coding: utf-8 -*-

"""
・WAを乗り越え自力AC
・条件ややこしくて色々バグらせるやつ。
・基本的にはS,Tそれぞれ反対側にいるやつで4つ組間では自由に移動できるけど、
　置換処理していいのがSだけなのが条件を複雑にしてる。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
S = input()
T = input()

A = [Counter() for i in range(N//2)]
B = [Counter() for i in range(N//2)]
for i in range(N//2):
    A[i][S[i]] += 1
    A[i][S[N-i-1]] += 1
    B[i][T[i]] += 1
    B[i][T[N-i-1]] += 1

def check(C):
    if len(C) == 1:
        return True
    if len(C) > 2:
        return False
    for v in C.values():
        if v != 2:
            return False
    return True

cnt = 0
for i in range(N//2):
    C1, C2 = A[i], B[i]
    C = C1 + C2
    if check(C):
        continue
    if len(C2) == 1:
        cnt += 1
        continue
    for k, v in C2.items():
        for _ in range(v):
            if C1[k] > 0:
                C1[k] -= 1
            else:
                cnt += 1
if N % 2 == 1 and S[N//2] != T[N//2]:
    cnt += 1
print(cnt)
