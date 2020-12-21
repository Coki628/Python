# -*- coding: utf-8 -*-

"""
解説見て式整理版
→値がでかくなりすぎてCE
"""

N = int(input())
aN = list(map(int, input().split()))

print(N+1)
# 後ろから順に、mod Nがいい感じに整うように加算してく
for i in range(N-1, -1, -1):
    # N * aN[i]でNの倍数(mod Nが0)にして、+ iで余る数調整
    # 現在値との差分にしたいので - aN[i]
    add_num = N * aN[i] + i - aN[i]
    print(1, i+1, add_num)
    for j in range(i+1):
        aN[j] += add_num
# 最後に全部mod Nすれば完了
print(2, N, N)
for i in range(N):
    aN[i] %= N
