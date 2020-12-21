# -*- coding: utf-8 -*-

"""
解説見て式整理版2
"""

N = int(input())
aN = list(map(int, input().split()))

print(N+1)
# 後ろから順に、mod Nがいい感じに整うように加算してく
for i in range(N-1, -1, -1):
    add_num = N*2 - aN[i] % N + i
    print(1, i+1, add_num)
    for j in range(i+1):
        aN[j] += add_num
# 最後に全部mod Nすれば完了
print(2, N, N)
for i in range(N):
    aN[i] %= N
