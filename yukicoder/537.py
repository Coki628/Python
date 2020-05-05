# -*- coding: utf-8 -*-

"""
・自力AC
・約数列挙
・ほぼ貼るだけかと思ったら、WAしてちょっと詰まった。
・対になってるやつが同じ文字列として重複しないかは考えたんだけど、
　対になってるやつ以外でも重複する可能性があるのに気づくのに時間かかった。
・最終的にsetで全体を管理して無事AC。
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

def divisors(N: int) -> set:
    """ 約数列挙 """
    from math import sqrt

    s = {1, N}
    for i in range(2, int(sqrt(N))+1):
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N = INT()

divs = sorted(divisors(N))
M = len(divs)
ans = set()
for i in range(ceil(M, 2)):
    j = M - i - 1
    s1 = str(divs[i]) + str(divs[j])
    s2 = str(divs[j]) + str(divs[i])
    ans.add(s1)
    ans.add(s2)
print(len(ans))
