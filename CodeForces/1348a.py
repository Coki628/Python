# -*- coding: utf-8 -*-

"""
・WAいっぱい踏んだ。。
・結果的に、一番大きいのを振った方と逆側に2番目以降大きい半分全部を振る、
　でとりあえず通った。。
・たしかに2冪だと大きい値はすごく大きくなるから、1番大きいのを振った方には
　なるべく他は小さいのを割り当てるのが自然だよね。。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = []
    for i in range(1, N+1):
        A.append(2**i)

    a = b = 0
    a += A.pop()
    for i in range(N//2):
        b += A.pop()
    a += sum(A)
    ans = abs(a - b)
    print(ans)
