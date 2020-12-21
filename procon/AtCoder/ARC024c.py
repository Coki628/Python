# -*- coding: utf-8 -*-

"""
・自力AC！
・文字列系、dictを工夫して出現済管理
・出現済管理はABC141eでの高速化と同じアプローチ。
・ただし今回はキーとして使うのが、要素26個のtupleだったので、高速に機能するか心配だった。
・毎回list→tuple変換しないといけなかったのも気になった。
・がしかし結局、速度は全然問題なくて、N<=10万で余裕のpythonAC0.2秒。
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

N, K = MAP()
# アルファベットを0〜25の数値に変換しておく
S = [ord(s)-97 for s in input()]

known = {}
cur = [0] * 26
for i, s in enumerate(S[:K]):
    cur[s] += 1
# 各文字毎の出現回数をキーとして、値に位置を持たせる
known[tuple(cur)] = K - 1
for i, s in enumerate(S[K:], K):
    # 先頭にあった文字を引いて、今回の文字を足す
    cur[S[i-K]] -= 1
    cur[s] += 1
    key = tuple(cur)
    if key in known:
        # 出現済の位置が、現在位置とK以上離れていればOK
        if i - known[key] >= K:
            YES()
            exit()
    else:
        # 未出現のキーなら今回の位置を持たせる
        known[key] = i
# 該当するのものが見つからなければNG
NO()
