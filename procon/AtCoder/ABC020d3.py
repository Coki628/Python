# -*- coding: utf-8 -*-

"""
参考：http://tsutaj.hatenablog.com/entry/2017/07/01/073648
　　　http://baitop.hatenadiary.jp/entry/2018/09/04/172911
　　　https://atcoder.jp/contests/abc020/submissions/6416586
　　　https://www.slideshare.net/chokudai/abc020
・GCD系、LCM
・式変形、等差数列、包除原理
・lcm(a, b) = a * b / gcd(a, b) を使ってLCMからGCDの式に変形。
・GCD毎にまとめて数え上げる。
・中の計算では数を数えるけど、この時に各素因数の個数は関係なくて、
　見るのは何種類の素因数を使ったか。それに応じて包除原理の足し引きを変える。
・足し引きする量は、その時の素因数を組み合わせたある値が、nまでに何回出てきたか。
　等差数列の和を使って総和を取る。
・計算量はめんどいから省略するけど、N,K共に10億でもpythonAC0.1秒かからなかった。
　約数列挙のループ中で全組み合わせとかやってるのにね。
　やっぱ約数列挙も素因数もかなり数は少なくなるってことだよね。
"""

import sys
from itertools import combinations

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

def factorize(num: int) -> dict:
    """ 素因数分解 """
    from math import sqrt
    from collections import Counter

    d = Counter()
    # 終点はルート切り捨て+1
    for i in range(2, int(sqrt(num))+1):
        # 素因数分解：小さい方から割れるだけ割って素数をカウント
        while num % i == 0:
            num //= i
            d[i] += 1
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumは素数(ただし1^1は1^0なので数に入れない)
    if num != 1:
        d[num] += 1
    return d

def divisors(N: int) -> set:
    """ 約数の列挙・個数 """
    from math import sqrt

    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て
    for i in range(2, int(sqrt(N))+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

# 等差数列の和：(初項a, 末項b, 項数c)
def get_sum(a, b, c):
    return (a+b) * c // 2

N, K = MAP()

# lcm(a, b) = a * b / gcd(a, b) の部分の変形と計算
def calc(n, k):
    # 素因数分解(今回は各素数が何個あったかは必要ない)
    primes = factorize(k).keys()
    # 1~nまでの総和(これをベースにしてここから足し引きしていく)
    a = (n+1) * n // 2
    b = K
    # 各素因数の全組み合わせ
    for i in range(1, len(primes)+1):
        for comb in combinations(primes, i):
            num = 1
            for prime in comb:
                num *= prime
            # カウントする個数はnまで
            cnt = n // num
            # 何個の素因数を使ったか → 何個の素因数の倍数として出てくるか
            # の偶奇によって足すか引くかする(包除原理)
            if i % 2 == 0:
                # 総和は等差数列の和で出せる
                a += get_sum(num, num*cnt, cnt)
            else:
                a -= get_sum(num, num*cnt, cnt)
    # 総和が集まったら、sum(a) * b / 1 を返却
    return a * b

ans = 0
# gcd(i, K)の答えはKの約数のどれかになるため、各約数毎に計算する
for div in divisors(K):
    # gcd(i, K) = div を、gcd(i/div, K/div) = 1 の形で考える
    ans += calc(N // div, K // div)
    ans %= MOD
print(ans)
