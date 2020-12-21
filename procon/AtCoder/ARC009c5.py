# -*- coding: utf-8 -*-

"""
参考：http://mayokoex.hatenablog.com/entry/2016/03/28/234940
・別解。この方針に辿り着けるなら、攪乱順列を知らなくても解ける。
・包除原理、数え上げ、組み合わせ
・行き先が合っていない手紙(バラバラ)は、N個中K個がバラバラ、はすぐに出ないけど、
　N個中K個以下がバラバラ、であれば組み合わせと階乗で簡単に表せる。
・例えばN個中3個以下バラバラなら、N個から3個選んでその3つは自由にシャッフルさせる。
　とすると、nC3*3!になって、これはN-3個は合ってるのが確定だけど、3個については、
　バラバラかもしれないし、合ってるかもしれない、の状態になるので、
　N個中3個"以下"バラバラの通り数になる。(参考及び自分ノートも参照)
・これが分かれば、包除原理でN個中0個バラバラまでを全部+-すれば、重複が消えてK個バラバラの通り数が出せる。
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
MOD = 1777777777

def nCr(n, r, MOD):
    """ 組み合わせの数(大きいnに対して定数回だけやる用) """

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

N, K = MAP()

# 包除原理で、k個全ての要素がバラバラ(ai != i)になる通り数を出す
def calc(k):
    # まずはk!をやる
    tmp = 1
    for i in range(1, k+1):
        tmp *= i
        tmp %= MOD
    # バラバラがk個以下(k-1個かもしれないし、0個かもしれないし、ようは全部)の集合はk!全通り
    res = tmp
    # バラバラがk-1個以下〜0個の集合について、包除原理で+-していく
    for i in range(1, k+1):
        tmp *= pow(i, MOD-2, MOD)
        tmp %= MOD
        if i % 2 == 1:
            res -= tmp
        else:
            res += tmp
        res %= MOD
    return res

print(nCr(N, K, MOD) * calc(K) % MOD)
