# -*- coding: utf-8 -*-

"""
ちょっと編集
・残り調べるとこ、ビットシフトにしてみた。この方が前からいけるしいいかな。
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
for i in range(19):
    if rest >> i & 1:
        num = 2 ** i
        # ちょうどいい位置(ここより手前で2**iまでを網羅)にバイパスを作る感じ
        edges.append([i+1, end, L-rest])
        rest -= num
print(v, len(edges))
for i in range(len(edges)):
    print(*edges[i])
