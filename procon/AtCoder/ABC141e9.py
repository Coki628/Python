# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2019/09/16/014600
　　　https://img.atcoder.jp/abc141/editorial.pdf
　　　https://snuke.hatenablog.com/entry/2014/12/03/214243
・Z-algorithmでLCP(最長共通接頭辞の長さ)を取得する。
・これも計算量はO(N^)=5000^2=2500万でpythonだとTLEでpypyAC0.7秒。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
S = input()

def Z_algorithm(S):
    N = len(S)
    res = [0] * N
    res[0] = N
    i = 1
    j = 0
    while i < N:
        while i+j < N and S[j] == S[i+j]:
            j += 1
        res[i] = j
        if j == 0:
            i += 1
            continue
        k = 1
        while i+k < N and k+res[k] < j:
            res[i+k] = res[k]
            k += 1
        i += k
        j -= k
    return res

ans = 0
for i in range(N):
    # Z-algorithmでLCP(最長共通接頭辞の長さ)を取得
    LCP = Z_algorithm(S[i:])
    for j, lcp in enumerate(LCP):
        # 最大値更新の際は、共通部分が重複していないことを確認する
        if lcp > ans and j >= lcp:
            ans = lcp
print(ans)
