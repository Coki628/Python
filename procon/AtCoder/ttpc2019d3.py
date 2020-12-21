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
・Grundy数の取得を、定義通りにDPするようにした。
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

primelist = eratosthenes_sieve(max(A))
primeset = set(primelist)

# 集合に含まれない最小の非負整数(minimum excluded)の取得
def get_mex(a):
    i = 0
    while True:
        if i not in a:
            return i
        i += 1

# dp[i] := 素数iのGrundy数
dp = {}
# aset[i] := 素数iの遷移先のGrundy数の集合(0はどこからでも行けるので最初から追加しておく)
aset = {prime: {0} for prime in primelist}

# 各素数についてこのゲームでのGrundy数を取得
for prime in primelist:
    # primeから行ける遷移先のGrundy数をasetに追加する
    if prime-2 in primeset:
        aset[prime].add(dp[prime-2])
        aset[prime].add(dp[2])
    # 遷移先の情報が揃ったら、mexから今回のprimeのGrundy数を取得
    dp[prime] = get_mex(aset[prime])

# 山毎のGrundy数
grundy = [0] * N
for i, a in enumerate(A):
    grundy[i] = dp[a]

# Grundy数の全xorが0以外なら勝ち
ans = reduce(xor, grundy, 0)
if ans != 0:
    print('An')
else:
    print('Ai')
