# -*- coding: utf-8 -*-

"""
参考：http://kmjp.hatenablog.jp/entry/2013/09/06/0900
　　　http://algoogle.hadrori.jp/tdpc/h/
　　　http://fluffyowl.hatenablog.com/entry/2017/09/15/200320
・蟻本演習2-3-14
・ソートしてからナップサックDP
・グループ毎にまとめてから処理、という点ではjoi2010final_bと共通してるけど、
　これは重さがあるから、それだけだとグループ内でどこから使うかが一意に定まらない。
・各状態から1色増やしたDPをループ内でやって、元の状態+1色との最大を取る、といったことをやる。
・C色以内のjループは後ろからやらないと更新順がうまくいかない。
・計算量はC^2*N*W=50^2*100*10000=25億？多分もっと少なくなってるんだと思う。
・numpy実装。pythonAC0.3秒。
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

N, W, C = MAP()
M = 50
colors = [[] for i in range(M)]
for i in range(N):
    w, v, c = MAP()
    # 色毎にまとめておく
    colors[c-1].append((w, v))

# dp[i][j] := i色以下、重さj以下で達成できる価値の最大値
dp = np.zeros((C+1, W+1), dtype=np.int64)
for i in range(M):
    # 後ろからやる
    for j in range(C-1, -1, -1):
        # 現在の状態のコピーを作る
        tmp = dp[j].copy()
        # 普通のナップザックをやる
        for w, v in colors[i]:
            if w <= W:
                prev = tmp.copy()
                tmp[w:] = np.maximum(tmp[w:], prev[:W+1-w]+v)
        # 元の状態+1色と今回の結果で最大を取る
        dp[j+1] = np.maximum(dp[j+1], tmp)
print(dp[C,W])
