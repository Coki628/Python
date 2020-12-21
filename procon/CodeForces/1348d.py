# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/76555
・公式解
・貪欲に最大限増やして、最後に端数をいい位置に差し込む。頭いい。。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()

    # C[i] := i日目のバクテリアの個数
    C = []
    k = 0
    sm = 0
    # まず最大限分裂させたものとして、終了(かその直前)まで配列を構築
    while sm + 2**k <= N:
        sm += 2**k
        C.append(2**k)
        k += 1
    # 足りない分は追加してソートすれば、ちょうどいい位置に収まる
    if sm < N:
        C.append(N-sm)
        C.sort()
    ans = []
    for i in range(len(C)-1):
        ans.append(C[i+1] - C[i])
    print(len(ans))
    print(*ans)
