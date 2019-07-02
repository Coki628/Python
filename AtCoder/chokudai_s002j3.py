# -*- coding: utf-8 -*-

"""
参考：https://scrapbox.io/snuke/Chokudai_SpeedRun_002
・想定解もやってみた。
・約数列挙久々に使った。
・確かにGCDって約数なんだからこいつと相性いいはずだよな。
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

def divisor_set(N: int) -> set:
    """ 約数の列挙・個数 """
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て
    for i in range(2, int(sqrt(N))+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N=INT()
a,b=MAP()

s=divisor_set(a)
s|=divisor_set(b)

AB=[]
for i in range(1, N):
    a,b=MAP()
    AB.append((a, b))

# 最初のa,bの約数列挙を大きい方から試していく
s=sorted(s, reverse=True)
for num in s:
    for a, b in AB:
        # aでもbでも割り切れないnumだったらそれはNGなので次へ
        if a%num!=0 and b%num!=0:
            break
    else:
        # 最後まで行けたらOK
        print(num)
        break
