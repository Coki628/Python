# -*- coding: utf-8 -*-

"""
・多少悩んだけど自力AC！
・数え上げ、重複組み合わせに帰着
・広義単調増加列の通り数をどう数えるか悩んだけど、j=iから始まる2重ループがnH2だった(ABC21dでやった)のを思い出して、
　今回も自分より先で自分より小さい数が使えないなら、それと条件は同じじゃないかと思ったのが当たった。
・結果的に、これならうまく778と877みたいのを全部同一視してくれるので、正しい数え上げができた。
・これがもし同値を許さない単調増加列だったら、今度はnHrじゃなくてnCrがうまくハマるんだと思う。
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

def nCr(n, r, MOD):
    """ 組み合わせの数(大きいnに対して使用する。計算量はr) """

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    num = den = 1
    # 分子の計算
    for i in range(n, n-r, -1):
        num *= i % MOD
        num %= MOD
    # 分母の計算
    for i in range(r, 0, -1):
        den *= i % MOD
        den %= MOD
    # logがつくため、MOD除算は最後の1回だけにする
    return num * pow(den, MOD-2, MOD) % MOD

def nHr(n, r, MOD):
    return nCr(r+n-1, r, MOD)

N = INT()
A = LIST()

# -1が連続する区間毎にまとめる(開始地点の値、終了地点の値、区間長)
B = []
unknown = False
for i, a in enumerate(A):
    if unknown and a != -1:
        t = A[i]
        ti = i - 1
        unknown = False
        B.append((s, t, ti-si))
    if not unknown and a == -1:
        s = A[i-1]
        si = i - 1
        unknown = True

ans = 1
for s, t, l in B:
    b = t - s + 1
    # 重複組み合わせ
    ans *= nHr(b, l, MOD)
    ans %= MOD
print(ans)
