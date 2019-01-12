# -*- coding: utf-8 -*-

"""
・元index保持して値でソートして戻したい時、
　実は元順に戻す必要ないっていう知見があったので改善版。
・あとindex付でリスト作り直すとこも内包表記で簡潔に。
"""

from operator import itemgetter
from collections import Counter

N, K = map(int, input().split())
aN = list(map(int, input().split()))
# この方が宣言と分けなくていいし簡潔
aN2 = [[i] + [aN[i]] for i in range(N)]

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
ans = [0] * N
for i, num in aN2:
    # 同じ数値が並んでいれば被らない
    # ans[元のidx]に直接答え入れれば、この後元順にソートする必要ない
    ans[i] = clr % K + 1
    # 色は1～Kで循環させる
    clr += 1
print('YES')
print(*ans)
