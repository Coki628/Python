"""
・さくっと自力AC！
・半分全列挙
・制約N<=40が解法を示していた。
　昔のABCのナップザックのやつからコード持ってきて、ちょっと改変する。
　価値がなくて重さだけみたいな感じになるのでこっちのが単純。
・今見返すと、番兵とか入れないでよく通ったな。。。
"""

import sys
from bisect import bisect_right

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

N, T = MAP()
A = LIST()

A1 = A[:N//2]
T1 = []
for i in range(1<<len(A1)):
    t = 0
    for j in range(len(A1)):
        if i>>j & 1 == 1:
            t += A1[j]
    if t <= T:
        T1.append(t)
T1.sort()

A2 = A[N//2:]
ans = 0
for i in range(1<<len(A2)):
    t2 = 0
    for j in range(len(A2)):
        if i>>j & 1 == 1:
            t2 += A2[j]
    if t2 <= T:
        t1 = T1[bisect_right(T1, T-t2)-1]
        ans = max(t1+t2, ans)
print(ans)
