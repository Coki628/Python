# -*- coding: utf-8 -*-

"""
・しっかり考察して自力AC。毎回こうだといい。
・でもこれで緑中位diffくらいなんだよね。もっと手早く解けるようにしないとな。。
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

N = INT()
A = LIST()

# とりあえず立ち位置に関する値は規則的に+1ずつなので、その分をまず平らにしておく
B = [0] * N
for i in range(N):
    B[i] = A[i] - i
C = Counter(B)
offset = 0
ans = 0
for i in range(N-1):
    # 自分以前の値はペアに使えないので引く
    C[B[i]] -= 1
    # 自分とペアになれる数を足す
    ans += C[-A[i]-offset]
    # 1つ進めば相対的に立ち位置は1つずれるので、これで調整
    offset += 1
print(ans)
