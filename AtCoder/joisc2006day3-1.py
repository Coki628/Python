# -*- coding: utf-8 -*-

"""
参考：http://kagamiz.hatenablog.com/entry/2012/11/06/005615
　　　https://examist.jp/mathematics/baainokazu/onajimono-jyunretu/
・自力ならず。。
・辞書順、同じものを含む順列
・前から順に、ソート済と比較しながら、自由に並べられる後ろの方を数え上げていく。
"""

import sys
from collections import Counter
from math import factorial

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

S = input()
N = len(S)

# 同じものを含む順列の計算：n!/p!q!r!
def calc(C):
    # n!
    res = factorial(sum(C.values()))
    # 各要素(p,q,r..)の階乗
    for v in C.values():
        res //= factorial(v)
    return res

S2 = sorted(S)
ans = 0
for i, s in enumerate(S):
    j = 0
    se = set()
    # この位置に入る文字より小さい間、その後ろを自由に並べられる
    while S2[j] < s:
        # 同じ文字を固定したら同じ組をもう1回数えてしまうのでやらない
        if S2[j] not in se:     
            C = Counter(S2)
            C[S2[j]] -= 1
            # 同じものを含む順列の計算
            ans += calc(C)
            se.add(S2[j])
        j += 1
    # sと一致した文字はこの位置に確定、次回以降使えなくなるので除く
    S2.pop(j)
# 最後に自分の分を足す
print(ans + 1)
