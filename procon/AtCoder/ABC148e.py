# -*- coding: utf-8 -*-

"""
・参戦中なんとか自力AC
・とはいえ色々いじってるうちに、5冪がなんか怪しいと気づいてエスパーしたようなもの。
・ちゃんと論理立てて理解したのはコンテスト終了後落ち着いて考えた後。
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

def f(n):
    if n <= 1:
        return 1
    else:
        return n * f(n-2)
    
# for i in range(100):
# for i in [90, 100, 140, 150, 160, 200, 1240, 1250, 1250, 1260]:
#     print(i, f(i))

N = INT()

A = []
i = 0
while 5 ** i <= 10 ** 18:
    A.append(5**i * 10)
    i += 1

if N % 2 == 0:
    ans = 0
    for a in A:
        ans += N // a
    print(ans)
else:
    print(0)
