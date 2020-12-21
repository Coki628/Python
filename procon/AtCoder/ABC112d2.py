# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！
・GCD系、約数列挙
・当時全然考察生えなかったぽいから、今回ちゃんと解けて良かった。
・GCDをxとするには、数列の全ての要素をxの倍数にする。
　そのためには、総和となるMがN個(必要な要素数)以上のxに分けれるかを確認すればいい。
　必要なのは約数列挙と、その約数中でMからN個以上作れるものの最大を見つけること。
　数列に余計な数が混ざるとGCDはすぐ1とかになるので、これが最善となる。
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

N, M = MAP()

# 約数列挙
divs = sorted(divisors(M), reverse=1)
# 約数を大きい方から見ていって、N個以上作れるならそれが最善
for div in divs:
    cnt = M // div
    if cnt >= N:
        print(div)
        exit()
