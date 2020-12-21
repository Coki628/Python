# -*- coding: utf-8 -*-

"""
参考：https://pitsbuffersolution.com/compro/atcoder/arc089d.php
　　　http://tutuz.hateblo.jp/entry/2018/06/20/212752
　　　https://paiza.hatenablog.com/entry/2014/05/28/%E3%82%82%E3%81%97%E5%A5%B3%E5%AD%90%E5%A4%A7%E7%94%9F%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9E%E3%81%AB%E3%80%8E%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0%E3%80%8F%E3%82%92%E5%9B%B3
　　　http://www.learning-algorithms.com/entry/2018/01/31/144519
　　　https://img.atcoder.jp/arc089/editorial.pdf
・2次元累積和
・0-indexedのまま行くか1-indexedにするか人によって色々で、
　それぞれに考慮する点があると思った。
・今度は0-indexedでループ減らした版。
"""

import sys
def input(): return sys.stdin.readline().strip()

N, K = map(int, input().split())

# ここは循環するK*2まで作る
B = [[0] * (K*2) for i in range(K*2)]
for i in range(N):
    x, y, c = input().split()
    x, y = int(x), int(y)
    # 白に塗る条件はxかyにKずらして黒の条件にする(これで2色考えなくていい)
    if c == 'W':
        y += K
    # 循環するのでmod K*2でやる
    B[x%(K*2)][y%(K*2)] += 1

# ここはK*2の正方形を右下まで動かすので、K*4まで作っておく
sm = [[0] * (K*4) for i in range(K*4)]
# 2次元累積和を取る
for i in range(K*4):
    for j in range(K*4):
        # K*2からはみ出る範囲も正しく累積和を出すため、K*2までの値を循環させる
        # 0-indexedなのでここはシンプル
        sm[i][j] = B[i%(K*2)][j%(K*2)]
        # 縦横の累積和を同時に集めていく
        # 0-indexedなので、i-1,j-1がマイナスになる場合を考慮
        if j > 0:
            sm[i][j] += sm[i][j-1]
        if i > 0:
            sm[i][j] += sm[i-1][j]
        if i > 0 and j > 0:
            sm[i][j] -= sm[i-1][j-1]

# 2次元累積和に必要な4点の加減を行う
def get_sum(x1, y1, x2, y2, sm):
    # 右下は必ず足す
    res = sm[x2][y2]
    # 左下を引く
    if x1 > 0:
        res -= sm[x1-1][y2]
    # 右上を引く
    if y1 > 0:
        res -= sm[x2][y1-1]
    # 左上を足す
    if x1 > 0 and y1 > 0:
        res += sm[x1-1][y1-1]
    return res

ans = 0
for i in range(K*2):
    for j in range(K*2):
        # 1つ目の正方形
        tmp = get_sum(i, j, i+K-1, j+K-1, sm)
        # 2つ目の正方形
        tmp += get_sum(i+K, j+K, i+K*2-1, j+K*2-1, sm)
        ans = max(tmp, ans)
print(ans)
