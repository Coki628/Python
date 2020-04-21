# -*- coding: utf-8 -*-

"""
・ちょうどいいとこまで内側whileで飛ばしてくやつ。
・Bがソート済なら、1つ前を済ませた続きから再開できるからN^2ループにならなくて済む。
　今回はBが元々ソート済で問題にも明記されてたからヒントになって優しい。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M = MAP()
A = LIST()
B = [b-1 for b in LIST()]

dom = 0
num = 0
cur = 0
ans = []
for b in B:
    while cur + A[dom] <= b:
        cur += A[dom]
        dom += 1
    num = b - cur
    ans.append((dom+1, num+1))
[print(a, b) for a, b in ans]
