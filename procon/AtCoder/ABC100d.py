# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc100/editorial.pdf
　　　http://drken1215.hatenablog.com/entry/2018/06/16/234000
・3項目をそれぞれ+するか-するか2^3通り試す。
・合計の絶対値は合計*-1なので、(値*-1)+(値*-1)..とやっても同じ結果になるのが分かる。
・DPっぽいと思ってしまったけど、各値の制約が10**10と大きいからそれはない
　っていう判断ができるようになること。
"""

N, M = map(int, input().split())
xyzN = [None] * N
for i in range(N):
    x, y, z = map(int, input().split())
    xyzN[i] = (x, y, z)

ans = 0
# 3項目を+か-にする2^3通りループ
for i in range(1 << 3):
    S = []
    for j in range(N):
        sm = 0
        # 3項目を個別に確認して+か-か決める
        for k in range(3):
            # ビット演算ですっきりまとまる
            if i >> k & 1:
                sm += xyzN[j][k]
            else:
                sm -= xyzN[j][k]
        # 各ケーキ毎に今回のパターンでの合計を格納していく
        S.append(sm)
    # 降順ソートしてMまでの合計
    S.sort(reverse=True)
    ans = max(sum(S[:M]), ans)
print(ans)
