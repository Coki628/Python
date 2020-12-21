# -*- coding: utf-8 -*-

"""
参考：https://yukicoder.me/problems/no/571/editorial
・こっちのが書くの楽だな。reverseもitemgetterも打たなくていいから速くなるし。
・降順ソートしたい時は符号逆にすれば、複数キー使いたい時も全部昇順扱いにできるので1度で済む。
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

AB = []
for i in range(3):
    a, b = MAP()
    AB.append((-a, b, chr(i+65)))

AB.sort()
for a, b, c in AB:
    print(c)
