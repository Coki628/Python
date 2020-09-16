"""
・すげー悩んだ。
・みんな通しまくってるから、適当な貪欲投げたら通った。。。
・具体的には、
　・動かせるところは全部降順に並べる。
　・この状態で累積和を見て、負になってしまう場所は無理なので、そこは前に回す。
　ってやった。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()
    B = LIST()

    A2 = []
    for i in range(N):
        if not B[i]:
            A2.append(A[i])
    A2.sort()

    A3 = []
    for i in range(N):
        if not B[i]:
            A3.append(A2.pop())
        else:
            A3.append(A[i])
    acc = list(accumulate(A3))
    A4 = []
    ans = [None] * N
    for i in range(N):
        if acc[i] < 0 and not B[i]:
            A4.append(A3[i])
        else:
            ans[i] = A3[i]
    j = 0
    for i in range(N):
        if ans[i] is None:
            ans[i] = A4[j]
            j += 1
    print(*ans)
