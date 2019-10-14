# -*- coding: utf-8 -*-

"""
・自力ならず。
・数学系、式変形
・xかyを決め打ちして、整数解になる所を探せばいい、は思ったけど、
　例えばyを決め打つ場合、引き分けは少ない方がいいはず(引き分けが多い方が試合数Nをオーバーしやすい)なので、
　0からループを回すとして、どこまでの値を取りうるかが分からなかった。
　他の人のを見る限り、yはwまでの値しか取らないようだけど、どうやってそれを導いたんだろう。。
　自分でも思いついたかもしれない条件として、x + y > nをつけてみたけどまだTLEで、
　というかyを決め打つ場合、x < 0にならないことも条件に必要で、これもテストケース見たから分かっただけで、
　自分で気づけた可能性は低い。
　で、その条件をつけて、yの上限が分からないことに関しては、時間オーバーしかけたら-1、
　みたいな方法を取ったらうまくいった。これでAC。
"""

import sys
import time

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
INF = float('inf')
MOD = 10 ** 9 + 7

n, p, w, d = MAP()

x = y = 0
start = time.time()
while True:
    x = (p-d*y) / w
    if x + y > n or x < 0:
        print(-1)
        break
    if (p-d*y) % w == 0:
        x = int(x)
        z =  n - x - y
        if z >= 0:
            print(x, y, z)
        else:
            print(-1)
        break
    y += 1
    elapsed_time = time.time() - start
    if elapsed_time > 0.7:
        print(-1)
        break
