# -*- coding: utf-8 -*-

"""
・終了後ちょっと整理。以下、自分なりに解釈した内容。
・末尾に0が付く → *10が必要
・奇数だとそもそも2がなくて1つも10が作れないので0確定。以下、偶数の場合。
・*10は冪で考えると5と2が1つずつ必要になるけど、今回掛けていく数の性質上、
　2はとても多くなるので5がいくつあるかだけ考えればOK。
・5の倍数なら1つ、5^2の倍数ならもう1つ、って感じで増える。
・でも今回偶数なら偶数しか使えないので、全部*2した時を考えればOK。
・後は5*2, 5^2*2, 5^3*2..の倍数がNにいくつ含まれるかを数えるため、N/aしていく。
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

A = []
i = 1
while 5**i * 2 <= INF:
    A.append(5**i * 2)
    i += 1

if N % 2 == 0:
    ans = 0
    for a in A:
        ans += N // a
    print(ans)
else:
    print(0)
