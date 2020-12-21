# -*- coding: utf-8 -*-

"""
参考：ツイート
・自力ならず。
・合成数、素因数分解
・問題文読んでたはずなのに、途中で要素が合成数のみっていう話が抜けてた。。
　まあそれを分かってたとして、正解に辿り着けたかは分からないけど。。
　出てくる素因数集めるところまでやって、11個より全然多い、どうやって11個に絞るんだ、
　うーんうーん、ってなってたからね。。
・1000以下の合成数は√1000=33以下の素数(これがちょうど11個)を約数に含む。ので、それらを使えばいい。
・知るかよ、、ってなるけど、まあしょうがない。。
　要素の種類が11個以下で、ってところから、小さい方から11個の素数で全部いけるんじゃないか、
　をエスパーできるべきだったのかなぁ。整数論学び直すしかないかねー。
"""

import sys
from collections import Counter

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

def factorize(num: int) -> dict:
    from math import sqrt
    from collections import Counter

    d = Counter()
    for i in range(2, int(sqrt(num))+1):
        while num % i == 0:
            num //= i
            d[i] += 1
        if num == 1:
            break
    if num != 1:
        d[num] += 1
    return d

for _ in range(INT()):
    N = INT()
    A = LIST()

    C = []
    for a in A:
        C.append(factorize(a))
    S = set()
    for c in C:
        S.add(min(c))
    ans = [0] * N
    used = [False] * len(S)
    idx = 0
    for i, p in enumerate(S):
        for j, c in enumerate(C):
            if ans[j] != 0:
                continue
            if p in c:
                if not used[i]:
                    idx += 1
                    used[i] = True
                ans[j] = idx
    print(idx)
    print(*ans)
