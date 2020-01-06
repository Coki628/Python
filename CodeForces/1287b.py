# -*- coding: utf-8 -*-

"""
・自力AC！
・3つ組を2つ決め打ちすると3つ目が定まる2重ループのやつ
・早めに方針に気付けて良かった。
・計算量が1500^2/2*30=3375万でpypyギリギリAC2.8秒(制約3秒)
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

N, K = MAP()
A = [input() for i in range(N)]

S = set(A)
D1 = {'S': 1, 'E': 2, 'T': 3}
D2 = {1: 'S', 2: 'E', 3: 'T'}

cnt = 0
for i in range(N):
    a1 = A[i]
    for j in range(i+1, N):
        a2 = A[j]
        a3 = []
        for k in range(K):
            if a1[k] == a2[k]:
                a3.append(a1[k])
            else:
                key = 6 - D1[a1[k]] - D1[a2[k]]
                a3.append(D2[key])
        if ''.join(a3) in S:
            cnt += 1
print(cnt // 3)
