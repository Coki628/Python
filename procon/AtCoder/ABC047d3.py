# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・出現済の値を管理しながら進んでいくやつ。
・でも制約でAは相異なるってあったから、もっと簡単にやれたね。問題文見れてないや。
"""

import sys

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

N, T = MAP()
A = LIST()

# 現在の状態での最大利益を求める
mn = INF
base = -INF
for a in A:
    base = max(base, a - mn)
    mn = min(mn, a)

# 最大を作れるペアがあったら潰していく
S = set()
ans = 0
for a in A:
    if a-base in S:
        ans += 1
        S.remove(a-base)
    S.add(a)
print(ans)
