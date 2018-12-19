# -*- coding: utf-8 -*-

"""
参考：https://minus9d.hatenablog.com/entry/2018/09/02/004610
・2べきで作るといいのは気づいたんだけどなぁ。
・2べき以外の数値をどうつなげばうまく処理できるかまとめられず…。
"""

L = int(input())

# 2べきで表せる基本部分を作る
edges = []
# baseは基本部分の守備範囲
v = base = 1
while base * 2 <= L:
    edges.append([v, v+1, 0])
    edges.append([v, v+1, base])
    base *= 2
    v += 1
# ゴールの頂点
end = edges[-1][1]
# 残りがいくつか
rest = L - base
# 残りを2べきに分割
for i in range(18, -1, -1):
    num = 2 ** i
    if rest >= num:
        # ちょうどいい位置(ここより手前で2**iまでを網羅)にバイパスを作る感じ
        edges.append([i+1, end, L-rest])
        rest -= num
print(v, len(edges))
for i in range(len(edges)):
    print(*edges[i])
