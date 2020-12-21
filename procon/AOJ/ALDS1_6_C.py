# -*- coding: utf-8 -*-

"""
クイックソート
"""

N = int(input())
aN = [[None] * 2 for i in range(N)]
for i in range(N):
    suit, num = input().split()
    aN[i][0] = suit
    aN[i][1] = int(num)

def partition(p, r):
    # 今回は基準xの選び方が最後尾固定だけど、
    # これをうまく選ぶ方がクイックソートは高速になる
    x = aN[r][1]
    i = p-1
    for j in range(p, r):
        if aN[j][1] <= x:
            i = i+1
            aN[i], aN[j] = aN[j], aN[i]
    aN[i+1], aN[r] = aN[r], aN[i+1]
    # 基準となった末尾要素のindex
    return i+1

def quick_sort(p, r):
    if p < r:
        # 再帰処理の中でパーティションを使う
        q = partition(p, r)
        quick_sort(p, q-1)
        quick_sort(q+1, r)

# 安定比較用にライブラリで安定ソートする
aN2 = sorted(aN, key=lambda x: x[1])
quick_sort(0, N-1)

# クイックソートは安定ではないが、
# マージソートに対して、メモリ領域を使わない利点がある
if all([aN[i][0] == aN2[i][0] for i in range(N)]):
    print('Stable')
else:
    print('Not stable')

for i in range(N):
    print(*aN[i])
