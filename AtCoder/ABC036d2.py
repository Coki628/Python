# -*- coding: utf-8 -*-

"""
参考：http://keita-matsushita.hatenablog.com/entry/2016/12/30/180918
　　　http://abc036.contest.atcoder.jp/data/abc/036/editorial.pdf
　　　https://atcoder.jp/contests/abc036/submissions/6034476
・木DP
・再帰探索(木の走査なのでメモはなしでOK)
・末端の葉から順に、戻ってくる時に答えを掛け合わせていく。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N=INT()
nodes=[[] for i in range(N)]
for i in range(N-1):
    a,b=MAP()
    nodes[a-1].append(b-1)
    nodes[b-1].append(a-1)

def rec(u, prev):
    wht=blk=1
    for v in nodes[u]:
        if v!=prev:
            res=rec(v, u)
            # 頂点uが黒ならその次(子)は白だけ
            blk*=res[0]
            # 頂点uが白ならその次(子)両方置ける
            wht*=(res[0]+res[1])
    return (wht%MOD, blk%MOD)

ans=rec(0, -1)
# 最後に白黒両方足せばOK
print((ans[0]+ans[1])%MOD)
