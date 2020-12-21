# -*- coding: utf-8 -*-

"""
参考：https://qnighy.github.io/informatics-olympiad/joi2011-day4-bookshelf-comment.html
　　　http://algoogle.hadrori.jp/joi/sc2011bookshelf/
　　　http://h0rnet.hatenablog.com/entry/2016/03/14/JOI_11_Day4_Bookshelf_(_DP_%2B_Segment_Tree_)
・蟻本演習3-3-1
・セグメント木
・動かすコストの最小値 = 全コスト - 動かさないコストの最大値
　から、答えを求める。でも言い換え考察力要るわー。
・pythonTLE,pypyMLE,メモリ制限64Mつらい。
・少しは軽くなるかと思ってセグ木の代わりにBIT使ったけど変化なし。。
・案外惜しいっぽいことが分かってきたので、不要マクロ外して、BITをクラスからメソッドに切り替えて再トライ。
・pythonAC0.9秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]

def init(n):
    # 0-indexed
    nv = 1
    while nv < n:
        nv *= 2
    size = nv
    tree = [0] * nv
    return size, tree
    
def query(i):
    """ [0, i]の値を取得 """
    s = 0
    i += 1
    while i > 0:
        s = max(s, tree[i-1])
        i -= i & -i
    return s
    
def update(i, x):
    """ 値の更新：添字i, 値x """
    i += 1
    while i <= size:
        tree[i-1] = max(tree[i-1], x)
        i += i & -i

N = INT()
cost = LIST(N)
A = [a-1 for a in LIST(N)]

# tree[A[i]] := 前からiまで見て、A[i]を動かさない時の、動かさない総コストの最大値
size, tree = init(N)
total = sum(cost)
mx = 0
for i in range(N):
    a = A[i]
    res = query(a-1)
    update(a, res+cost[a])
mx = query(N-1)
# 全て動かす場合 - 動かさない総コストの最大 = 動かすコストの最小
print((total-mx)*2)
