# -*- coding: utf-8 -*-

"""
・これもだいぶ悩んだ。
・制約から、答えになる数列は元と比べてだいぶ大きくできることが分かる。
　で、元のやつ1つに対してありうる数全部を含めるくらいに伸ばしていい、
　ってなったら、後は微調整で通った。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K = MAP()
    A = LIST()

    S = set(A)
    M = len(S)
    # 種類数が多すぎたらダメ
    if M > K:
        print(-1)
        continue
    # 種類数がKに満たない部分は適当に埋める
    seg = list(S) + [1]*(K-M)
    ans = seg * N
    print(len(ans))
    print(*ans)
