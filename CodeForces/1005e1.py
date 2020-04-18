# -*- coding: utf-8 -*-

"""
・ここまで自力AC！結構頑張ったと思う。
・2値化して累積和して集計
・左端を固定して、ペア組んだ時に中央値Mを達成できる右端がいくつあるか見る方針で行く。
・詰まるところ、Mより大きいか小さいかしか大事じゃないので、それで+1か-1に2値化しておく。(Mは0)
・Mの場所は常に決まっているので、その手前と先で扱いを変える。
・手前はMより大小がいくつずつあるかを持っておいて、左端が進むに合わせて更新。
・先は累積和したものを値毎に集計しておいて、左の大小合計 + 右の大小合計 が0か1になるものを数える。
・式変形して 右合計 = 1 - 左合計 or -左合計 にして、さっき集計した右合計を使って足す。
"""

import sys
from itertools import accumulate
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

N, M = MAP()
A = LIST()

for i in range(N):
    if A[i] == M:
        A[i] = 0
        idx = i
    elif A[i] < M:
        A[i] = -1
    else:
        A[i] = 1
A1 = A[:idx+1]
A2 = A[idx:]
acc = list(accumulate(A2))
C = Counter(acc)
sm = sum(A1)
ans = 0
for a in A1:
    ans += C[1-sm]
    ans += C[-sm]
    sm -= a
print(ans)
