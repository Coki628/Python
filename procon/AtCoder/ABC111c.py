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

# 一番多い数字が同じだった場合は別処理(内容は自信ない)
while max(even, key=even.get) == max(odd, key=odd.get):

    if max(even.values()) >= max(odd.values()):
        odd.pop(max(odd, key=odd.get))
    else:
        even.pop(max(even, key=even.get))
    if len(odd) == 0 or len(even) == 0:
        print(N // 2)
        exit()

print(N - max(even.values()) - max(odd.values()))