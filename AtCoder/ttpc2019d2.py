# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/ttpc2019/editorial.pdf
　　　https://www.creativ.xyz/grundy-number-1065/
・Grundy数、Nim応用、素数列挙
・実は自分で考えてたターン数がほぼGrundy数だった。
・各山をGrundy数にすれば、通常のNimと同じ解法に帰着できる。(XORで勝敗が分かる)
・今回はGrundy数=最大残りターン数で合ってるけど、場合によっては違うから気を付ける。
　(例えば遷移先が0,1,3なら4じゃなくて2、mexって言う)
・計算量について、素数列挙は100万でも全然間に合う。(pythonAC0.4秒)
"""

import sys
from operator import xor
from functools import reduce

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
INF = float('inf')
MOD = 10 ** 9 + 7

def eratosthenes_sieve(n):
    """ 素数列挙(エラトステネスの篩) """

    table = [0] * (n + 1)
    prime_list = []
    for i in range(2, n + 1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i + i, n + 1, i):
                table[j] = 1
    return prime_list

N = INT()
A = LIST()

primeset = set(eratosthenes_sieve(max(A)))

# 山毎に使えるターン数
B = [0] * N
for i in range(N):
    B[i] = 1
    j = 2
    # -2してまだ素数ならターン数を増やせる
    while A[i]-j in primeset:
        B[i] += 1
        j += 2

# Grundy数の全xorが0以外なら勝ち
ans = reduce(xor, B, 0)
if ans != 0:
    print('An')
else:
    print('Ai')
