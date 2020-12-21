"""
・自力AC！
・ソートして、累積和して、にぶたん。典型組み合わせでこういうのは好き。時間かかって悔しい。
・まず前提で、ソートして隣同士を組ませるのが最善となる。生徒の並びが決まるので、後は先生を
　どの値として挿入するかを全部見る。この時に先生が入る位置によって、その位置より後ろで
　ペアになるところが1つずつずれるので、偶奇で累積和を2種類取っておいて、
　これを前後適切な位置までのものを各回で使う。添字バグるのつらい。
"""

import sys
from itertools import accumulate
from bisect import bisect_left

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, M = MAP()
A = LIST()
B = LIST()

A.sort()
acc1 = [0] * (N+1)
acc2 = [0] * (N+1)
for i in range(N-1):
    if i % 2 == 0:
        acc1[i+1] = A[i+1] - A[i]
    else:
        acc2[i+1] = A[i+1] - A[i]
acc1 = list(accumulate(acc1))
acc2 = list(accumulate(acc2))

ans = INF
for b in B:
    idx = bisect_left(A, b)
    if idx == N:
        res = acc1[idx-1] + abs(b-A[idx-1])
    elif idx % 2 == 1:
        res = acc1[idx-1] + abs(b-A[idx-1]) + (acc2[-1]-acc2[idx])
    else:
        res = acc1[idx] + abs(b-A[idx]) + (acc2[-1]-acc2[idx])
    ans = min(res, ans)
print(ans)
