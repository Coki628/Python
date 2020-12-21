# -*- coding: utf-8 -*-

"""
参考：http://mayokoex.hatenablog.com/entry/2016/03/22/201438
・KMP法の動作確認
・文字列100万だけど、pythonでもAC0.5秒。
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

def KMP(S: str, T: str) -> list:
    """ KMP法(Sの中にTとマッチする場所のindexのリストを返す)：O(|S|+|T|) """

    N = len(S)
    M = len(T)
    # KMPをやるための前計算テーブルを構築
    table = [0] * (M+1)
    table[0] = -1
    j = -1
    for i in range(M):
        while j >= 0 and T[i] != T[j]:
            j = table[j]
        j += 1
        table[i+1] = j
    # KMP法
    res = []
    m = i = 0
    while m+i < N:
        if T[i] == S[m+i]:
            i += 1
            if i == M:
                res.append(m)
                m = m + i - table[i]
                i = table[i]
        else:
            m = m + i - table[i]
            if i > 0:
                i = table[i]
    return res

N = INT()
M = INT()
S = input()

T = 'IO' * N + 'I'
res = KMP(S, T)
print(len(res))
