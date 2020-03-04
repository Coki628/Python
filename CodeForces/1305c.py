# -*- coding: utf-8 -*-

"""
・自力ならず。。ツイート参考でAC。
・鳩の巣原理
・まず、mod取って同値になる2要素があると、他がどうであれ答えは0。(そもそもこれに気付けなかった。)
・そうでない場合は20万のNに2重ループ回すことになるけど、要素数が多い場合は絶対どこかで重複が出る。
・よって実際にループ回すのは要素数が少ない場合だけになるので問題なく計算できる。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, mod = MAP()
A = LIST()

# 1つでも重複があれば0 (鳩の巣原理より、大きいNは全てこっちになる)
modA = [a%mod for a in A]
if len(modA) != len(set(modA)):
    print(0)
    exit()

# 重複がなかった場合は普通に2重ループ回す
ans = 1
for i in range(N):
    for j in range(i+1, N):
        ans *= abs(A[i] - A[j])
        ans %= mod
print(ans)
