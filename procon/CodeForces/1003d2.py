# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1003/submission/66657630
・C++でlog2つ付いてても何とかAC通せたけど、みんなlog1つにしてたのでやってみる。
・軸にする方が逆だった。クエリで来た数の各2冪に対して集計した数のうちどれが使えるかを見ていく方針でなく、
　集計した数を見て、それを使ってクエリで来た数をどう減らせるか見ていく。(多分説明下手)
・こっちだと、集計したやつクエリ毎に元に戻さなきゃ、とかも考えなくてよくなる。
・自分より下のビットにはどうせ使えないので、上のビットから見て、
　自分より上のビットを含めて最大何回使えるのかは q // k すれば良かった。(ここが自分からは出ない)
"""

import sys
from collections import Counter

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, Q = MAP()
A = LIST()

C = sorted(Counter(A).items(), reverse=1)
out = []
for _ in range(Q):
    q = INT()

    ans = 0
    for k, v in C:
        cnt = min(q // k, v)
        q -= cnt * k
        ans += cnt
    
    if q > 0:
        out.append(str(-1))
    else:
        out.append(str(ans))
print('\n'.join(out))
