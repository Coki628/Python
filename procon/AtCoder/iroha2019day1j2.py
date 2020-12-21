# -*- coding: utf-8 -*-

"""
参考：https://www.hamayanhamayan.com/entry/2019/05/01/101449
　　　http://kmjp.hatenablog.jp/entry/2019/04/30/0930
・回文、転倒数
・回文だから半分決まれば全部決まる。
・まずは簡単だから偶数のやつから作る。
・式変形(2次方程式だから解は2つ)、詳細は参考1にて。
・転倒数がKになる0(1)の個数が分かったら後はその組み合わせの数。
・nCrをMOD付の大きい数で使うので、事前テーブル準備。
"""

import sys
from math import sqrt

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def init_fact_inv(MAX: int, MOD: int) -> list:
    """
    階乗たくさん使う時用のテーブル準備
    MAX：階乗に使う数値の最大以上まで作る
    """
    MAX += 1
    # 階乗テーブル
    factorial = [1] * MAX
    factorial[0] = factorial[1] = 1
    for i in range(2, MAX):
        factorial[i] = factorial[i-1] * i % MOD
    # 階乗の逆元テーブル
    inverse = [1] * MAX
    # powに第三引数入れると冪乗のmod付計算を高速にやってくれる
    inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
    for i in range(MAX-2, 0, -1):
        # 最後から戻っていくこのループならMAX回powするより処理が速い
        inverse[i] = inverse[i+1] * (i+1) % MOD
    return factorial, inverse

# 組み合わせの数(必要な階乗と逆元のテーブルを事前に作っておく)
def nCr(n, r, factorial, inverse):
    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    # 分子の計算
    numerator = factorial[n]
    # 分母の計算
    denominator = inverse[r] * inverse[n-r] % MOD
    return numerator * denominator % MOD

# 2次方程式の解を返却する(いわゆる解の公式)
def dim2formula(a, b, c):
    try:
        return (-b+sqrt(pow(b, 2)-4*a*c))/(2*a), (-b-sqrt(pow(b, 2)-4*a*c))/(2*a)
    # 解なしはNoneを返却
    except ValueError:
        return None, None

fact,inv=init_fact_inv(10**5+1, MOD)

Q=INT()
for _ in range(Q):
    N,K=MAP()
    # まずは偶奇
    if N%2==0:
        comb=comb1=comb2=0
        ans1,ans2=dim2formula(-2, 2*(N//2), -K)
        # ansが0以上の整数になっていればOK
        if ans1 is not None and ans1>=0 and ans1.is_integer():
            comb1=nCr(N//2, int(ans1), fact, inv)
        if ans2 is not None and ans2>=0 and ans2.is_integer():
            comb2=nCr(N//2, int(ans2), fact, inv)
        # 解が1つか2つか
        if ans1==ans2:
            comb+=comb1
        else:
            comb+=comb1+comb2
    else:
        # 奇数で真ん中が1
        comb=comb1=comb2=0
        ans1,ans2=dim2formula(-2, 2*(N//2)+1, -K)
        if ans1 is not None and ans1>=0 and ans1.is_integer():
            comb1=nCr(N//2, int(ans1), fact, inv)
        if ans2 is not None and ans2>=0 and ans2.is_integer():
            comb2=nCr(N//2, int(ans2), fact, inv)
        if ans1==ans2:
            comb+=comb1
        else:
            comb+=comb1+comb2
        # 奇数で真ん中が0
        comb1=comb2=0
        ans1,ans2=dim2formula(-2, 2*(N//2)-1, N//2-K)
        if ans1 is not None and ans1>=0 and ans1.is_integer():
            comb1=nCr(N//2, int(ans1), fact, inv)
        if ans2 is not None and ans2>=0 and ans2.is_integer():
            comb2=nCr(N//2, int(ans2), fact, inv)
        if ans1==ans2:
            comb+=comb1
        else:
            comb+=comb1+comb2
    print(comb%MOD)
