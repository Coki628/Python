# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/jsc2019-final/editorial.pdf
　　　https://tokumini.hatenablog.com/entry/2019/09/30/130000
・公式解
・鳩の巣原理
・最初、解説見ても何を言ってるのか分からなかった。
　いや、だから全ペア見ようとしたら10万のNが2乗でTLE確定じゃん、って。
・別の記事も参考によくよく考えたら、分かった。値の取りうる範囲が約200万までと決まっている以上、
　数が多い時は全ペア見る前に絶対どこかで重複が出る。だからそれを見つければいい。と。
・仕組みが分かってしまえば実装は確かに軽い。300点。でもこの考察は300点じゃないよなー。
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

N, M = MAP()
A = LIST()
B = LIST()

C = [()] * (max(A)+max(B)+1)
for i, a in enumerate(A):
    for j, b in enumerate(B):
        if C[a+b]:
            i2, j2 = C[a+b]
            print(i, j, i2, j2)
            exit()
        else:
            C[a+b] = (i, j)
print(-1)
