# -*- coding: utf-8 -*-

"""
・自力AC
・区分求積法
・放物線の面積、こんな方法で求められるだな。
　とりあえずxを10万個に区切って長方形の面積総和したらちゃんとACした。
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

a, b = MAP()

if a == b:
    print(0)
    exit()

each = (b - a) / 10**5
x = a
ans = 0
while x <= b:
    y = (x-a) * (x-b)
    ans += each * -y
    x += each
print(ans)