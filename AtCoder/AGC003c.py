# -*- coding: utf-8 -*-

"""
・600点自力AC！
・ちょっと考察詰めきれなくて自信なかったけど、考え方はほぼ想定解通りでよかった。
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

N = INT()
A = LIST(N)
A2 = sorted(A)

# 元の列とソート済それぞれの位置の偶奇を集める
A_mod = {}
A2_mod = {}
for i in range(N):
    A_mod[A[i]] = i % 2
    A2_mod[A2[i]] = i % 2

cnt = 0
for k, v in A_mod.items():
    # 位置の偶奇の合わない所は操作1を使うことになる
    if v != A2_mod[k]:
        cnt += 1
print(cnt//2)
