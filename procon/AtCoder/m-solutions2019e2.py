# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/m-solutions2019/editorial.pdf
　　　https://atcoder.jp/contests/m-solutions2019/submissions/5733769
　　　https://atcoder.jp/contests/m-solutions2019/submissions/5747263
・階乗MODの事前テーブル作る系
・公差1なら階乗だよなーは思ったけど、全項を公差で割れば公差1になるは出なかったなー。。
"""

import sys

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
MOD = 10 ** 6 + 3

# 階乗たくさん使う時用のテーブル準備
# MAX：階乗に使う数値の最大以上まで作る
def init_fact_inv(MAX: int, MOD: int) -> list:
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

fact, inv = init_fact_inv(MOD, MOD)

Q=INT()
for i in range(Q):
    x,d,n=MAP()

    # 例外処理
    if x==0:
        print(0)
        continue
    if d==0:
        print(pow(x, n, MOD))
        continue
    
    invd=pow(d, MOD-2, MOD)
    l=(x*invd-1)%MOD
    r=l+n

    # 例外処理(MODで割り切れてしまうケース)
    if r>=MOD:
        print(0)
        continue

    print(fact[r]*inv[l]*pow(d, n, MOD)%MOD)
