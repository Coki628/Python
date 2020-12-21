# -*- coding: utf-8 -*-

from operator import itemgetter
from collections import Counter

N, K = map(int, input().split())
aN = list(map(int, input().split()))
aN2 = [[0] * 3 for i in range(N)]
for i in range(N):
    aN2[i][0] = i
    aN2[i][1] = aN[i]

d = Counter(aN)
# 色が足りない
if d.most_common()[0][1] > K:
    print('NO')
    exit()
# 人が足りない
if len(aN) < K:
    print('NO')
    exit()

# 数値でソート
aN2.sort(key=itemgetter(1))
clr = 0
for i in range(N):
    # 同じ数値が並んでいれば被らない
    aN2[i][2] = clr % K + 1
    # 色は1～Kで循環させる
    clr += 1

# 元のindexでソート
aN2.sort(key=itemgetter(0))
ans = [0] * N
for i in range(N):
    ans[i] = aN2[i][2]
print('YES')
print(*ans)
