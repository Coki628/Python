"""
参考：https://codeforces.com/contest/1355/submission/80339731
　　　https://twitter.com/_shop_one/status/1261652828474363910
・自力ならず。。これは解けるべき問題だったな。。
・三角形の性質、式変形、前計算、いもす応用、累積和
・三角形は一番長い辺が短い2辺の和より小さければ成立するのでそれを使う。
　具体的にはx+y>zになるような組み合わせを全て数える。(ここまでは分かったんだが。。)
・いもすを上手に使うと、a+b全ペアのカウントが線形でいける。
　辺xをaiに固定すると、辺yは[b,c]の範囲なので、x+yは区間[ai+b,ai+c]を取りうる。
　これを全てのaiについてやる。
・該当するzは大きい値を全て使えるので、後ろから累積和する。
・x+y>=zではなくx+y>zなので+1する。
・D雰囲気で解けちゃったからレートは溶けなかったけど、本来こういう典型組み合わせみたいなのを
　確実に通せるようになるべき。修行が足りんなー。。
"""

import sys
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

a, b, c, d = MAP()
MAX = 10**6 + 7

imos = [0] * MAX
# 辺xをaiに固定すると、辺yは[b,c]の範囲なので、x+yは区間[ai+b,ai+c]を取りうる
# これを全てのaiについてやる
for ai in range(a, b+1):
    imos[ai+b] += 1
    imos[ai+c+1] -= 1
imos = list(accumulate(imos))
# 該当するzは大きい値を全て使えるので、後ろから累積和する。
acc = list(accumulate(imos[::-1]))[::-1]
ans = 0
for z in range(c, d+1):
    # x+y>=zではなくx+y>zなので+1する
    ans += acc[z+1]
print(ans)
