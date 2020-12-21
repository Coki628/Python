# -*- coding: utf-8 -*-

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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST()
B = LIST()

C1 = Counter()
C2 = Counter(B)
ans = INF
for i in range(N):
    a = A[i]
    C1[a] += 1
    # ここまでで使う、数列Aの中にあるaの数
    acnt = C1[a]
    # トータルの、数列Bの中にあるaの数
    bcnt = C2[a]
    # aについて、獲得可能なスコア
    cnt = bcnt // acnt
    # 今回のスコアは、今まで(の全部)か今回の小さい方
    ans = min(ans, cnt)
    print(ans)
