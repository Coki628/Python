"""
・ゲーム、素因数分解、場合分け
・基本的に奇数で割れれば割るし、割れなくて-1をやると奇数になるから負ける。
・ただし-1をやる時が2だった場合は1ももらった方が負けなので、そこは場合分け。
・奇数の素因数が1個しかないと先手は行動が限られるが、
　2個以上あればどうにでも減らす量を調節できるので、基本的に勝てる。
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
INF = 10 ** 19
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

    if N == 1:
        print('FastestFinger')
    elif N == 2:
        print('Ashishgup')
    else:
        fact = factorize(N)
        cnt = 0
        for k, v in fact.items():
            if k % 2 == 1:
                cnt += v
        if cnt == 1 and fact[2] == 1:
            print('FastestFinger')
        elif cnt >= 1:
            print('Ashishgup')
        else:
            print('FastestFinger')
