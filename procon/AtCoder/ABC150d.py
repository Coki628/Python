# -*- coding: utf-8 -*-

"""
・終了後なんとか自力AC
・GCD,LCM系
・LCMで割って奇数になるなら、その奇数を/2すれば半分LCMを~.5で作れるな、と言う方針は悪くなかった。
・がしかしその後色々迷走した。
・まず10万数列へのLCMが何故かTLE、計算量はGCDと同じlogNくらいだろうから、
　そんなに遅いはずないと思いつつもしばらく原因分からず。
　結果的に、LCMは限りなく巨大になるので、範囲外になった時点で切る必要があった。
・それでも後1つのWAが取れない。手元で色々試したりした結果、M境界ギリギリが該当する時に
　数えられてなかった。切り上げを切り捨て+1にすることで正しい答えが作れた。
"""

import sys
from fractions import gcd

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

def lcm(x, y): return (x * y) // gcd(x, y)

N, M = MAP()
A = LIST()

l = 1
for a in A:
    l = lcm(l, a)
    # LCMがM*2より大きくなったらもう範囲外
    if l > M*2:
        print(0)
        exit()

for a in A:
    # ここが偶数だと*~.5が作れない
    if l // a % 2 != 1:
        print(0)
        exit()
# 切り上げだと上限ギリが入る時WAするので切り捨て+1
ans = M*2 // l + 1
print(ans // 2)
