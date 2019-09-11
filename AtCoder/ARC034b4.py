# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc034
・公式解
・式変形、桁和系
・実はf(x)の取りうる範囲はかなり限定的なので、そこを全探索する。
・こんな簡単でよかったのかぁ。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

def f(x):
    x = str(x)
    res = 0
    for s in x:
        res += int(s)
    return res

N = INT()

# f(x)の最大値(9999...)
MAX = f(10**18-1)
ans = []
for x in range(max(0, N-MAX), N+1):
    # x = N - f(x) が成り立てば答えに追加
    if x == N - f(x):
        ans.append(x)
ans.sort()

print(len(ans))
[print(x) for x in ans]
