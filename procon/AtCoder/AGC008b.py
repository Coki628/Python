# -*- coding: utf-8 -*-

"""
・結構すんなり自力AC！
・累積和
・実は1つずらして操作することで、固定になる部分と自由に変えられる部分を確定できる。
　この考え方はABC083dでのものとほぼ同じ感じ。
・具体的には長さKの部分は白黒どっちかに固定で、それ意外の左右は自由に変えられる。
　ので、長さKをどこで取るかを全部見ればその最大値が答え。
・こういう数列ゴニョゴニョ系は適正範囲でも負けるのが結構あるので、ちゃんと考察できて嬉しい。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST()

# 自由に変えられる部分用の数列B(負の部分は取りたくないので0のまま)
B = [0] * N
for i, a in enumerate(A):
    if a > 0:
        B[i] = a

# 数列A,Bを累積和しておく
acc = [0] + list(accumulate(A))
acc2 = [0] + list(accumulate(B))
# Bは逆順も作っておくとこのあと楽
acc2rev = [0] + list(accumulate(B[::-1]))[::-1] + [0]

ans = -INF
for i in range(1, N-K+2):
    # 白黒固定部分を取るか取らないか
    cnt = max(acc[i+K-1] - acc[i-1], 0)
    # 自由に選べる範囲はBの累積和が最善になってる
    cnt += acc2[i-1] + acc2rev[i+K]
    ans = max(ans, cnt)
print(ans)
