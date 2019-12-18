# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/72243
　　　https://qiita.com/drken/items/ae02240cd1f8edfc86fd
・中国剰余定理から、60で割った余りが0 = 3で割った余りが0かつ20で割った余りが0
・3の倍数は桁和が3の倍数
・20の倍数は末尾が00,20,40,80のどれか
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

def check1(a):
    sm = 0
    for d in a:
        sm += int(d)
    return sm % 3 == 0

def check2(a):
    return a.count('0')

def check3(a):
    sm = 0
    for d in ['0', '2', '4', '6', '8']:
        sm += a.count(d)
    return sm >= 2

for _ in range(INT()):
    a = input()
    if check1(a) and check2(a) and check3(a):
        print('red')
    else:
        print('cyan')
