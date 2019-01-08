# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc106/editorial.pdf
　　　https://atcoder.jp/contests/abc106/submissions/3024983
　　　https://imoz.jp/algorithms/imos_method.html
　　　https://paiza.hatenablog.com/entry/2014/05/28/%E3%82%82%E3%81%97%E5%A5%B3%E5%AD%90%E5%A4%A7%E7%94%9F%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9E%E3%81%AB%E3%80%8E%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0%E3%80%8F%E3%82%92%E5%9B%B3
・二次元累積和で高速化：O(N^2+Q)
・最後にどこを足し引きするかは上記参考の二次元いもすと同じ話だと思う。
　→ちょっと違った。上記paizaの二次元累積和の説明がすごく参考になる。
"""

N, M, Q = map(int, input().split())
LR = [[0] * (N+1) for i in range(N+1)]
for i in range(M):
    l, r = map(int, input().split())
    LR[l][r] += 1

# 縦横両方について累積和を出す
# 先頭1行1列は0のままにしておく
for i in range(1, N+1):
    for j in range(2, N+1):
        LR[i][j] += LR[i][j-1]
for i in range(1, N+1):
    for j in range(2, N+1):
        LR[j][i] += LR[j-1][i]

for i in range(Q):
    p, q = map(int, input().split())
    # +左上 -右上 -左下 +右下
    ans = LR[p-1][p-1] - LR[p-1][q] - LR[q][p-1] + LR[q][q]
    print(ans)
