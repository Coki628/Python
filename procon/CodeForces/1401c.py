"""
・ここまで自力AC。(DもC++でAC)
・GCD、スワップ、ソート
・GCD取って最小値になるのは、最小値とGCD取って最小値になるやつだけ。(多分)
・これに該当する箇所だけを自由にスワップできるので、このグループ内だけソート。
・その順で元の数列に戻して、ソート済と一致してればOK、してなきゃNG。
"""

import sys
from math import gcd

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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()

    movable = [0] * N
    mn = min(A)
    B = []
    for i in range(N):
        if gcd(A[i], mn) == mn:
            movable[i] = 1
            B.append(A[i])
    B.sort()
    j = 0
    ans = A[:]
    for i in range(N):
        if movable[i]:
            ans[i] = B[j]
            j += 1
    if sorted(A) == ans:
        YES()
    else:
        NO()
