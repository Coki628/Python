# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc020
・GCD系、LCM
・式変形、等差数列
・lcm(a, b) = a*b/gcd(a, b) を使ってLCMからGCDの式に変形。
　gcd(a, b) = gcd(a%b, b) を使ってmod a毎に各値をまとめる。
　これでa*b/gcd(a, b)について、b(元々K固定)に加えてgcd(a, b)が固定になって、
　aも等差数列の和になるので、各mod aの和がO(1)で求まる。
・これでもまだ部分点解。。まあ満点解はまたそのうち。。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd

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
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()

d = N // K
ans = 0
for i in range(K):
    _gcd = gcd(i, K)
    mn = i
    # 末項と項数は末項候補がN以下かどうかで場合分け
    if d*K+i <= N:
        mx = d*K + i
        cnt = d + 1
    else:
        mx = d*K - K + i
        cnt = d
    # 式を整理すると、等差数列の和 * K / gcd になる
    ans += (mn+mx) * cnt // 2 * K // _gcd
    ans %= MOD
print(ans)
