"""
・自力ならず。。
・GCD、辞書順最大
・色々考えてしまったが、よくよく見たらO(N^2)通る制約だった。愚直いけるじゃん、、で終了後AC。。
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
INF = 10**19
MOD = 10**9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()

    b = c = max(A)
    ans = [b]
    A.pop(A.index(b))
    for _ in range(N-1):
        g = 0
        for a in A:
            if gcd(c, a) > g:
                g = gcd(c, a)
                b = a
        c = g
        ans.append(b)
        A.pop(A.index(b))
    print(*ans)
