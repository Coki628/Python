# -*- coding: utf-8 -*-

N = int(input())
vN = list(map(int, input().split()))

even = {}
odd = {}
# 奇数番目と偶数番目に分けて集計
for i in range(N):
    if i % 2 == 0:
        even[vN[i]] = even.get(vN[i], 0) + 1
    else:
        odd[vN[i]] = odd.get(vN[i], 0) + 1

# 集計数の多い順ソート
even_sorted = sorted(even.items(), key=lambda x: x[1], reverse=True)
odd_sorted = sorted(odd.items(), key=lambda x: x[1], reverse=True)

# 1番多い数字が同じだった場合の別処理
if even_sorted[0][0] == odd_sorted[0][0]:
    # 偶奇それぞれ数字が1種類しかない場合は、この後の比較処理のためにダミーの0を入れておく
    if len(even_sorted) == 1:
        even_sorted.append((0, 0))
    if len(odd_sorted) == 1:
        odd_sorted.append((0, 0))
    # 偶奇それぞれを2番目にしてみて、数の大きい方
    print(N - max(even_sorted[1][1] + odd_sorted[0][1], even_sorted[0][1] + odd_sorted[1][1]))
else:
    print(N - even_sorted[0][1] - odd_sorted[0][1])