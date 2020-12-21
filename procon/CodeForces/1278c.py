# -*- coding: utf-8 -*-

"""
・自力AC
・累積和、dictを工夫。
"""

import sys
from collections import Counter, defaultdict
from itertools import accumulate

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

for _ in range(INT()):
    N = INT()
    # それぞれ-1と1にする
    A = [a-2 if a == 1 else a-1 for a in LIST()]
    C = Counter(A)
    diff = abs(C[-1] - C[1])
    if diff == 0:
        print(0)
        continue
    # 左右の区間を分ける
    B = A[N:]
    A = A[:N][::-1]
    # 累積和
    A = [0] + list(accumulate(A))
    B = [0] + list(accumulate(B))
    # 苺がa個多い、ブルーベリーがa個多い、が最初に来る位置を持たせる
    D1 = defaultdict(lambda: INF)
    D2 = defaultdict(lambda: INF)
    for i in range(N, -1, -1):
        D1[A[i]] = i
        D2[B[i]] = i
    ans = INF
    for i in range(diff+1):
        # 左にi個、右にdiff-i個多く取る所まで進む
        a = i
        b = diff - i
        if C[-1] - C[1] > 0:
            a = -a
            b = -b
        idx1 = D1[a]
        idx2 = D2[b]
        ans = min(ans, idx1 + idx2)
    print(ans)
