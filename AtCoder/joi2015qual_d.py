# -*- coding: utf-8 -*-

"""
・自力AC、二分探索
・JOI予選の4問目は部分点続きだったから、ちゃんと通せたのはまあ嬉しい。
"""

import sys
from bisect import bisect_left

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

N, T, Q = MAP()
A = []
for i in range(N):
    a, d = MAP()
    A.append((a, d))
X = [x-1 for x in LIST(Q)]

# 人々が集まる位置を確認
B = []
for i in range(N-1):
    # 向かい合う位置の真ん中に集まる
    if A[i][1] == 1 and A[i+1][1] == 2:
        B.append((A[i][0] + A[i+1][0]) // 2)
# 左に進む人を処理しやすいように逆向きにしたリストを作成
Brev = sorted([-b for b in B])

ans = [0] * N
for i in range(N):
    a, d = A[i]
    # 右に向かう人
    if d == 1:
        idx1 = bisect_left(B, a)
        idx2 = bisect_left(B, a+T)
        # T秒後もidxが同じなら、止まらず進む
        if idx1 == idx2:
            ans[i] = a + T
        # 異なれば、集まる場所に止まる
        else:
            ans[i] = B[idx1]
    # 左に向かう人
    else:
        # 逆向きリストを使えば、二分探索がほぼ同じ形のまま使える
        a = -a
        idx1 = bisect_left(Brev, a)
        idx2 = bisect_left(Brev, a+T)
        if idx1 == idx2:
            ans[i] = -(a + T)
        else:
            ans[i] = -Brev[idx1]
[print(ans[x]) for x in X]
