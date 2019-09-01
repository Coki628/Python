# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2019/02/26/101100
・300点自力ならず。
・こういうのどうやったら気付くんだろう。ずるい。。
・再帰処理を遡る。
・終了条件に達するまでは、戻り値Pは+1で来ているので、これは潜った回数とイコールになるはず。
　であればこの回数、必要な処理をしてやれば、P=1000の時のNから1つずつ遡れる。
・P=1000の時、N=1789997546303が分かっているので、
　こいつに実際に1000回再帰させて、必要なPまで処理が戻って来た時に出力する。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

P = INT()

def f(x):
    if x >= 10**16 or x <= 1:
        res = 0
    elif x%2 == 0:
        res = f(x//2) + 1
    else:
        res = f(3*x+1) + 1
    # 目的のPまで処理が戻って来たら結果出力
    if res == P:
        print(x)
        exit()
    return res

# P=1000になるf(N)を実行
f(1789997546303)
