# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc010/editorial.pdf
・差分で作った数列で考える。で、操作回数を引く。
・すると、1ずつ増加していた場所は全て0に均されて、境界になっていた場所だけが出るので、
　どこを起点に何回操作するといいか分かる。
・考察つらー。でも、隣り合う差分に注目ってのは、見ておくべき視点だよなー。
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

N = INT()
A = LIST()

total = sum(A)
each = N*(N+1) // 2
if total % each != 0:
    NO()
    exit()
# 操作回数
K = total // each

# 隣り合うAの各値の差分で数列Bを作る
B = [0] * N
for i in range(N):
    j = (i+1) % N
    B[i] = A[j] - A[i]
# ここからK引けば、1度の操作につきBのいずれかの要素を+N、とできる
B = [b-K for b in B]

if all(b % N == 0 and b <= 0 for b in B):
    YES()
else:
    NO()
