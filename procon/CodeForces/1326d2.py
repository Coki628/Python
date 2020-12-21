# -*- coding: utf-8 -*-

"""
参考：https://snuke.hatenablog.com/entry/2014/12/02/235837
　　　https://twitter.com/titia_til/status/1240688581238075393
・自力ではない。ツイート確認して色々調べてAC。
・最長回文、Manacher
・聞いたことあったけど初めて使った。
・偶数長のものを考えた時に色々面倒があるので、分けて処理した。
・返されるリストからどの部分が使われるかは分かるんだけど、
　連結に妥当かどうか判定するのにindexの微調整が大変だった。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def Manacher(S):
    """ Manacher(文字iを中心とする最長の回文の半径を返す)：O(N) """

    N = len(S)
    R = [0] * N
    i = j = 0
    while i < N:
        while i-j >= 0 and i+j < N and S[i-j] == S[i+j]:
            j += 1
        R[i] = j
        k = 1
        while i-k >= 0 and i+k < N and k+R[i-k] < j:
            R[i+k] = R[i-k]
            k += 1
        i += k
        j -= k
    return R

def Manacher_even(_S):
    """ Manacher偶数長(文字iとi+1の間を中心とする最長の回文の半径を返す)：O(N) """

    _N = len(_S)
    S = []
    for i in range(_N-1):
        S.append(_S[i])
        S.append('$')
    S.append(_S[-1])
    N = len(S)
    R = [0] * N
    i = j = 0
    while i < N:
        while i-j >= 0 and i+j < N and S[i-j] == S[i+j]:
            j += 1
        R[i] = j
        k = 1
        while i-k >= 0 and i+k < N and k+R[i-k] < j:
            R[i+k] = R[i-k]
            k += 1
        i += k
        j -= k
    res = [0] * (_N-1)
    j = 0
    for i in range(1, N, 2):
        res[j] = R[i] // 2
        j += 1
    return res

for _ in range(INT()):
    S = input()
    N = len(S)
    i = 0
    j = N - 1
    T1 = []
    while i < j:
        if S[i] == S[j]:
            T1.append(S[i])
            i += 1
            j -= 1
        else:
            break
    else:
        print(S)
        continue
    T3 = T1[::-1]
    R = Manacher(S[i:j+1])
    mx = a = b = 0
    for k, r in enumerate(R):
        k += i
        ln = r * 2 - 1
        if k-r+1 == i:
            if ln > mx:
                mx = ln
                a, b = i, i+ln
        if k+r-1 == j:
            if ln > mx:
                mx = ln
                a, b = j+1-ln, j+1
    R = Manacher_even(S[i:j+1])
    for k, r in enumerate(R):
        k += i + 1
        ln = r * 2
        if k-r == i:
            if ln > mx:
                mx = ln
                a, b = i, i+ln
        if k+r-1 == j:
            if ln > mx:
                mx = ln
                a, b = j+1-ln, j+1
    T2 = list(S[a:b])
    ans = T1 + T2 + T3
    print(''.join(ans))
