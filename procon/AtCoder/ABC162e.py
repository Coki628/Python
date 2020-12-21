# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc162/editorial.pdf
・時間外に粘って考えたけど自力ならず。
　考え方は近いとこまで来てたのにな。。
・数え上げ、後ろから決めて行く。
・ある数xかその倍数がgcdになる通り数は(K//x)^N、までは出たんだけどな。
・内側のwhileで2,4,8,16って行くならlog分かるんだけど、
　2,4,6,8って行くのに計算量logKで済むのがなんか非自明。2乗の半分くらいかかりそう。
　結局それで諦めてしまった。ダメ元でも試せばよかったな。
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

N, K = MAP()

ans = 0
A = [0] * (K+1)
for x in range(K, 1, -1):
    A[x] = pow(K//x, N, MOD)
    i = x
    while i + x <= K:
        i += x
        A[x] -= A[i]
    A[x] %= MOD
    ans += A[x] * x
    ans %= MOD
ans += pow(K, N, MOD) - sum(A)
ans %= MOD
print(ans)
