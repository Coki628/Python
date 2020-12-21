# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc085/editorial.pdf
・公式解、O(1)の一発解法
・思いつけばいいけど、そういつも上手くいかないよね…。
"""
 
N, Z, W = map(int, input().split())
aN = list(map(int, input().split()))

if N == 1:
    print(abs(aN[0] - W))
else:
    # Xが最後まで引くか、最後から2番目まで引くか
    print(max(abs(aN[-1] - W), abs(aN[-2] - aN[-1])))
