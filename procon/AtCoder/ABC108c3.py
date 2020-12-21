# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/09/02/011000
・速解き特訓ABC周回
・自力ならず。
・記事調べて思い出した。これ当時も全然解けなくてモヤっとしたやつだ。
・やっぱり今見ても、公式解よりこっちの全探索する方のがまだしっくり来る。
・Kの倍数の話なので、まずMOD Kで考える。
・実はaだけ決め打ちすればb,cは求まる。b,cはMOD Kで同値になる。(まじか)
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, K = MAP()

# MOD Kでの値を集計
C = Counter()
for a in range(1, N+1):
    C[a%K] += 1

# aを決め打ちして全探索
ans = 0
for a in range(K):
    b = (K - a) % K
    c = (K - a) % K
    if (b + c) % K == 0:
        ans += C[a] * C[b] * C[c]
print(ans)
