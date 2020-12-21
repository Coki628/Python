# -*- coding: utf-8 -*-

"""
O(N^2/2)がTLE確定なのは分かってるけどとりあえずやる
"""

N, M = map(int, input().split())
aN = list(map(int, input().split()))

cnt = 0
for i in range(N):
    ans = 0
    for j in range(i, N):
        ans += aN[j]
        if ans % M == 0:
            cnt += 1
print(cnt)
