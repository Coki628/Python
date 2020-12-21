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
・定数倍高速化のためループまとめて(4K^2)*3回を1回にしたから、
　5200万がO((4K)^2+(2K)^2)=2000万くらいになって結構速くなったはず。
　…と思ったらメモリがオーバーしたから余計な配列減らした。
"""

import sys
def input(): return sys.stdin.readline().strip()

N, K = map(int, input().split())

# ここは循環するK*2まで作る
B = [[0] * (K*2+1) for i in range(K*2+1)]
for i in range(N):
    x, y, c = input().split()
    x, y = int(x), int(y)
    # 白に塗る条件はxかyにKずらして黒の条件にする(これで2色考えなくていい)
    if c == 'W':
        y += K
    # 循環するのでmod K*2でやる
    # 累積和でひとつ前を見に行くので、1-indexedにする
    B[x%(K*2)+1][y%(K*2)+1] += 1

# ここはK*2の正方形を右下まで動かすので、K*4まで作っておく
sm = [[0] * (K*4+1) for i in range(K*4+1)]
# 2次元累積和を取る
# 1-indexedなので、気にせず1からループすればいい
for i in range(1, K*4+1):
    for j in range(1, K*4+1):
        # K*2からはみ出る範囲も正しく累積和を出すため、K*2までの値を循環させる
        # 1-indexedにしているため、ここの加減がややこしい
        sm[i][j] = B[(i-1)%(K*2)+1][(j-1)%(K*2)+1]
        # 縦横の累積和を同時に集めていく
        sm[i][j] += sm[i][j-1] + sm[i-1][j] - sm[i-1][j-1]

# 2次元累積和に必要な4点の加減を行う
def get_sum(x1, y1, x2, y2, sm):
    # 1-indexedなので、x, yがマイナスの値になる考慮は必要ない
    res = sm[x2][y2] - sm[x1-1][y2] - sm[x2][y1-1] + sm[x1-1][y1-1]
    return res

ans = 0
for i in range(1, K*2+1):
    for j in range(1, K*2+1):
        # 1つ目の黒い正方形
        tmp = get_sum(i, j, i+K-1, j+K-1, sm)
        # 2つ目の黒い正方形
        tmp += get_sum(i+K, j+K, i+K*2-1, j+K*2-1, sm)
        ans = max(tmp, ans)
print(ans)
