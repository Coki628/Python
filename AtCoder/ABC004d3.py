# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc004
　　　https://atcoder.jp/contests/abc004/submissions/5384294
・メモ化再帰
・置かない場合の遷移をしない条件をうまく作れなくて、
　結局人の見ないと書けなかった。
・考察はともかく、このくらい自力実装できないとなぁ。。
・計算量は座標位置の数*マーブルの総数=1000*900=90万、python0.3秒AC。
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

R, G, B = MAP()
total = R + G + B

# どのマーブルを使うか
def calc(cur, cnt):
    # R
    if total - cnt > G + B:
        return abs(400-cur)
    # G
    elif total - cnt > B:
        return abs(500-cur)
    # B
    else:
        return abs(600-cur)

# memo[i][j] := マーブル使用回数jの時の、位置iから先の最小操作回数
memo = list2d(1001, total+1, INF)
def rec(cur, cnt):
    if memo[cur][cnt] != INF: 
        return memo[cur][cnt]
    # 使い切ったのでこの先のコストは0確定
    if cnt == total:
        memo[cur][cnt] = 0
        return 0
    # 現在位置curにマーブルを置く場合の遷移
    res = rec(cur+1, cnt+1)+calc(cur, cnt)
    # 残り移動回数 == マーブル残数 になると、もう置くしかない
    if 1000 - cur == total - cnt:
        memo[cur][cnt] = res
        return res
    # 大丈夫なら、マーブルを置かない場合の遷移と比較する
    res = min(res, rec(cur+1, cnt))
    memo[cur][cnt] = res
    return res

print(rec(0, 0))
