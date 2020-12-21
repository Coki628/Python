# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/camp/2015/2015-sp-tasks/2015-sp-d1-enjoi-review.pdf
・自力ならず。。
・累積和、再帰探索
・実はある始点からの全区間の確認は再帰10回程度で済むので、100万箇所やっても大丈夫。
・pypyでも1秒きつくて、C++でAC0.2秒。
・powの前計算、試しにこっちでもやってみたら、通ってるケースで0.8→0.6秒くらいには速くなったので、
　pythonでもある程度の効果があると思ってよさそう。まあそれでもTLEなんだけどね。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

K = INT()
N = 4 ** K
S = input()
pow4 = [0] * 11
for i in range(11):
    pow4[i] = pow(4, i)

S += S
acc = list2d(3, N*2+1, 0)
for i, s in enumerate(S):
    if s == 'J':
        acc[0][i+1] = 1
    elif s == 'O':
        acc[1][i+1] = 1
    else:
        acc[2][i+1] = 1
for i in range(3):
    acc[i] = list(accumulate(acc[i]))

def rec(i, k):
    if k == -1:
        return 1
    cnt = 0
    for j in range(3):
        l = i + pow4[k] * j
        r = i + pow4[k] * (j+1)
        cnt += acc[j][r-1] - acc[j][l-1]
    return cnt + rec(i+pow4[k]*3, k-1)

ans = 0
for i in range(1, N+1):
    ans = max(ans, rec(i, K-1))
print(N - ans)
