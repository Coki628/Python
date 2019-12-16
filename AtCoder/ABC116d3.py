# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力ならず。。
・うまくまとめられなかった。
・先に種類数を最大にしておいて、そこから種類数減ってもいいからポイント高いやつと交換、
　ってやったら、K個以内とかの兼ね合いがややこしくなって無限にバグらせて死んだ。
・逆が良かった。先にポイント高いやつK個持っておいて、そこからポイント減ってもいいから種類数増やす、
　の方針が正解だった。これだと種類数が増えない交換は全て確定で損、とみなすことができて、
　かなり候補が絞り込める。
・色々試行錯誤した結果、(公式解で言及されていた)優先度付きキューも要らなかった。
　交換候補がまとまった時点でまとめてソートするだけで、後で追加する用とかもない。
・各種類のうち最大のものは絶対交換に出さないので別にして保管、他を交換候補とする。
・実は交換候補の方のtが具体的に何なのかは、2つ目以降であることは分かっているので管理しなくてもいい。
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
DT = []
tmp = [[] for i in range(N)]
for i in range(N):
    t, d = MAP()
    DT.append((d, t))
DT.sort(reverse=1)

A = []
D = {}
dcnt = 0
# ポイント上位K個をまず取る
for i in range(K):
    d, t = DT[i]
    dcnt += d
    # 種類tが1つ目なら、確定で残す
    if t not in D:
        D[t] = d
    # 2つ目以降なら、最大のものを残して他は交換候補へ
    else:
        if D[t] < d:
            D[t], d = d, D[t]
        A.append(d)
A.sort(reverse=1)
tcnt = len(D)

ans = dcnt + tcnt**2
for i in range(K, N):
    d, t = DT[i]
    # 既に持っている種類ならスキップ
    if t in D:
        continue
    D[t] = d
    # 全部1種類ずつ(交換候補がもうない)なら、もう種類数を増やせないので終了
    if not A:
        break
    d2 = A.pop()
    # この候補と交換してみる
    dcnt += d
    dcnt -= d2
    tcnt += 1
    ans = max(ans, dcnt + tcnt**2)
print(ans)
