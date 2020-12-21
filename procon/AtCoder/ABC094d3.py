# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc095/editorial.pdf
・rは半分あたりがいいってのは分かってたのに…
　nが最大固定でいいって思えなかった。
・nが小さくなってもr次第で大きくなるかもって思ってしまったけど、
　大きかったnが使えなかったrは小さくなったnも使えないから、
　結局nは一番大きいのが最適だね。気付きたかった…。
"""

N = int(input())
aN = list(map(int, input().split()))
aN.sort(reverse=True)

n = aN[0]
r = aN[1]
for i in range(2, N):
    if abs(n / 2 - aN[i]) < abs(n / 2 - r):
        r = aN[i]
print(n, r)
