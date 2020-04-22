# -*- coding: utf-8 -*-

"""
・自力AC！
・再帰探索、バックトラック
・予め使える数値は集計しておいて、行けるなら進んで、ダメになったら元に戻す。
・遷移は各状態から2つだけで、それも該当する値がないと行けないので、そんなに多くはならない。
　開始地点として全ての場所を試せる。うまくいったら再帰から戻ってくる道すがらで
　答えになる値を集めてくる。最後にリバースして出力でOK。
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

def rec(a, d):
    if d == N:
        ans.append(a)
        return True
    if a % 3 == 0 and C[a//3]:
        C[a//3] -= 1
        if rec(a//3, d+1):
            ans.append(a)
            return True
        C[a//3] += 1
    if C[a*2]:
        C[a*2] -= 1
        if rec(a*2, d+1):
            ans.append(a)
            return True
        C[a*2] += 1
    return False

N = INT()
A = LIST()

C = Counter(A)
for i, a in enumerate(A):
    ans = []
    if rec(a, 1):
        print(*ans[::-1])
        exit()
