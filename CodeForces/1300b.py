# -*- coding: utf-8 -*-

"""
・WA。ちょっと問題読み違えてた。
　なんとなく、奇数人に分けるに当たって常に半分にできる生徒数が与えられる気がしてたんだけど、
　そんなことはなくて4人とかの場合もあった。(この場合、1,3に振り分けないといけない)
　このコードだと常に半分に分けちゃうので、2,2になってしまい、真ん中を取ったつもりが真ん中切り上げ
　になってしまいバグる。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()

    A.sort()
    B1 = []
    B2 = []
    for i in range(N*2):
        if i % 2 == 0:
            B1.append(A[i])
        else:
            B2.append(A[i])
    ans = abs(B1[N//2] - B2[N//2])
    print(ans)
