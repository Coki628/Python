# -*- coding: utf-8 -*-

"""
参考：https://naoyat.hatenablog.jp/entry/2012/01/04/231801
　　　https://nodchip.hatenadiary.org/entry/20090303/1236058357
・とりあえず何とか自力AC(参考はあくまで三分探索の実装方法のため)
・初めての三分探索
・あとちゃんと問題文から式立てれて良かった。
・ちょっと式変形版。
　1/(2**..)を0.5**..にしたら、計算中に値が大きくなり過ぎないので、
　上限値に自信がなくても適当に大きい値から始められる。
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
MOD = 10 ** 9 + 7

P = float(input())

def calc(x):
    return x + 0.5**(x/1.5) * P

# 三分探索
low = 0
hi = 10**18
for i in range(1000):
    m1 = (low*2+hi) / 3
    m2 = (low+hi*2) / 3
    res1 = calc(m1)
    res2 = calc(m2)
    # 今回は下に凸な関数なので値の小さい方に向かって狭めていく
    if res1 <= res2:
        hi = m2
    else:
        low = m1

x = m1
print(calc(x))
