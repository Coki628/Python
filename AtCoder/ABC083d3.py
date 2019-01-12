# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc088/editorial.pdf
・一番真ん中に近い境界の、始点か終点の遠い方の距離が最適
・中央付近から順に境界を1つずつ減らして最終的に全0にできる。
"""

S = input()

dist = float('inf')
for i in range(1, len(S)):
    if S[i] != S[i-1]:
        # 境界を見つけたら、始点か終点の遠い方の距離を取る
        tmp = max(i, len(S)-i)
        # この値が小さい程中央に近い位置
        if tmp < dist:
            dist = tmp
# 境界が1つもないなら、文字列の長さをそのまま使える
if dist == float('inf'):
    dist = len(S)

print(dist)
