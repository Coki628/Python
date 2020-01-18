# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=PYuhUgx6yU0&feature=youtu.be
・自力ならず。。
・二分探索、2値化、ビット全探索
・にぶたんして2値化するまでは行ったんだけど、2値になっても結局N^2ループじゃん、、
　でDPとかに迷走して終わった。。
・確かに数列30万全部見るより2^8を全ペア見る方が全然早い。
　N<=8でビット全探索とか、そりゃ考えるべきだったよなぁ。。
・計算量はにぶたんlogの内側でN*M+2^2Mだから、(240万+65000)*30くらいで約7400万。
　pypyAC3.4秒(制約5秒)
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

def bisearch_max(mn, mx, func):

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

def check(m):
    ok = [0] * N
    S = set()
    for i in range(N):
        for j in range(M):
            if A[i][j] >= m:
                ok[i] |= 1<<j
        S.add(ok[i])
    full = (1<<M) - 1
    for bit1 in range(1<<M):
        for bit2 in range(bit1, 1<<M):
            if bit1 in S and bit2 in S:
                if bit1 | bit2 == full:
                    return True
    return False

N, M = MAP()
A = [None] * N
for i in range(N):
    A[i] = LIST()

res = bisearch_max(0, 10**9+1, check)
ok = [0] * N
S = set()
D = {}
for i in range(N):
    for j in range(M):
        if A[i][j] >= res:
            ok[i] |= 1<<j
    S.add(ok[i])
    D[ok[i]] = i + 1
full = (1<<M) - 1
for bit1 in range(1<<M):
    for bit2 in range(bit1, 1<<M):
        if bit1 in S and bit2 in S:
            if bit1 | bit2 == full:
                print(D[bit1], D[bit2])
                exit()
