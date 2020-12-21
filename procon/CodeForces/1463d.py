"""
・自力ならず。。悔しい。
・貪欲、二分探索
・最大ケースを貪欲はやったんだ。でも合わなくて、方針変えてしまった。
　最小ケースの方でも貪欲やって、その間は全部OKってすればよかった。。
"""

import sys
from bisect import bisect_right

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
    B = LIST()
    C = [0] * (N*2+1)
    for b in B:
        C[b] += 1
    A = []
    for i in range(1, N*2+1):
        if not C[i]:
            A.append(i)

    i = 0
    cnt1 = 0
    for b in B:
        i = bisect_right(A, b, lo=i) + 1
        if i > N:
            break
        cnt1 += 1
    i = 0
    cnt2 = 0
    for a in A:
        i = bisect_right(B, a, lo=i) + 1
        if i > N:
            break
        cnt2 += 1
    r = cnt1
    l = (N-cnt2)
    ans = r - l + 1
    print(ans)
