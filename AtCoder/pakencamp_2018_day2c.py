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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

# 例外処理
s = set(A)
if 0 in s:
    s.remove(0)
if len(s) <= 1:
    print(1)
    exit()

# 周期mで構築ができるかどうか
def check(m):
    # 作業しやすいように配列の長さを周期と合わせる
    add = N % m
    B = A.copy() + [0]*add

    # 確定になる場所を埋める
    mod = [0] * m
    for i in range(N+add):
        if B[i] != 0:
            mod[i%m] = B[i]
    
    for i in range(N+add):
        # 確定した色と異なる箇所が1つでもあればNG
        if B[i] != 0 and B[i] != mod[i%m]:
            return False
    # 全部うまくいけばOK
    return True

for i in range(2, N+1):
    if check(i):
        print(i)
        break
