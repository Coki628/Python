# -*- coding: utf-8 -*-

"""
・色々工夫してなんとか自力AC
・BITで取ってた区間和は、動的に累積和を作りながら区間和取ってくるやつで対応。
・それでも今度はpythonがMLEしたので、よく見たらDP配列は前の方全く参照してないので
　1つの変数で答えだけ保持。これでpythonAC1.0秒。(制約1.5秒)
・始点を作るためにN+1してたところを累積和の先頭番兵のためにさらにN+2にしたので、
　indexが訳分からんくなって脳が崩壊しかけた。危なかったー。
"""

import sys
from bisect import bisect_left
from itertools import accumulate

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
MOD = 1234567

N, K = MAP()
A = [0] + list(accumulate(LIST(N)))

acc = [0] * (N+2)
acc[1] = ans = 1
for i, a in enumerate(A[1:], 1):
    # どこまで前から跳んでこれるか
    prev = bisect_left(A, max(a-K, 0))
    # 届く範囲全てから通り数を集めてくる
    ans = (acc[i] - acc[prev]) % MOD
    acc[i+1] = acc[i] + ans
print(ans)
