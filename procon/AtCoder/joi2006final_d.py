# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2006/2007-ho-prob_and_sol/2007-ho-review.pdf
・蟻本演習2-3-12
・DAG、メモ化再帰
・試合結果の勝敗を各チームを結ぶ有向辺としてDAGを構築する。
・この問題の文面からDAGに帰着できる発想が得られるようになりたいね。
・書いてみれば割と普通のメモ化再帰なんだけどね。。
・最初、python,pypy共にMLE。。この頃？メモリ制限64Mなんだよね。。
・で、5000^2=2500万の配列をnumpyで作ってメモリ節約。pythonAC0.4秒。
"""

import sys
import numpy as np

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
M = INT()
table = np.zeros((N, N), dtype=np.bool)
# nodes[i] := チームiに勝ったチームのリスト
nodes = [[] for i in range(N)]
for i in range(M):
    # a:win, b:lose
    a, b = MAP()
    table[a-1,b-1] = 1
    nodes[b-1].append(a-1)

memo = [0] * N
ranking = []
def rec(a):
    if memo[a]:
        return
    memo[a] = 1
    for b in nodes[a]:
        rec(b)
    # 一番最後にある頂点ほど順位が上位になる
    ranking.append(a)
    return

# 各勝敗の依存性から、メモ化再帰で順位を確定する
for i in range(N):
    rec(i)
[print(a+1) for a in ranking]

# 順位表の複数判定
for i in range(N-1):
    a, b = ranking[i], ranking[i+1]
    # 隣り合うチームの勝敗で未確定が1つでもあれば変更可
    if not table[a,b]:
        print(1)
        exit()
# なければ不可
print(0)
