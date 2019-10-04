# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-5、700点自力AC！これは嬉しい。
・内側で部分和DPするんだけど、メイン部分は木の問題。
・割り当てる値が明らかな、葉から処理していく。
・実は自分の子の色は親が自由に決められる(反転させれば影響はない)ことへの気付き、大事。
・あとは部分和取る時、値の種類が多くなりすぎて計算遅くならないか心配だったけど大丈夫だった。
　(N=1000だけど、各値が5000まで取りうるやつの和)
"""

import sys
from bisect import bisect_right

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
P = [p-1 for p in LIST()]
X = LIST()

nodes = [[] for i in range(N)]
for i, p in enumerate(P):
    nodes[p].append(i+1)

# W[(i, j)] := (自分を根とした木の自分側の色の重み, 反対側の色の重み)
W = [None] * N
def calc(u):
    # 葉はどうするか最初から決まってる
    if not nodes[u]:
        return (X[u], 0)

    A = []
    total = 0
    for v in nodes[u]:
        A.append(W[v])
        total += W[v][0] + W[v][1]
    # 自分の子の色をどうするかは自由に決められるはずなので、
    # 部分和を列挙して、一番いい(親に戻す重みが少なくて済むような)値を使う
    n = len(A)
    dp = [set() for i in range(n+1)]
    dp[0].add(0)
    for i in range(n):
        a, b = A[i]
        for s in dp[i]:
            dp[i+1].add(s+a)
            dp[i+1].add(s+b)
    B = sorted(dp[n])
    idx = bisect_right(B, X[u]) - 1
    # 自分の予定量より大きい重みしかなければNG
    if idx == -1:
        print('IMPOSSIBLE')
        exit()
    a, b = X[u], total - B[idx]
    return (a, b)

def rec(u, prev):
    for v in nodes[u]:
        rec(v, u)
    # 末端から順番に処理していく
    W[u] = calc(u)
rec(0, -1)
# 最後まで行けばOK
print('POSSIBLE')
