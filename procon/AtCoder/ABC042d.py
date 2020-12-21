# -*- coding: utf-8 -*-

from itertools import combinations as comb

H,W,A,B = map(int, input().split())

h = H - A
w = B + 1
ans = 0
# マスを右上に1つずつずらして、必ず通る場所でパターンを足し合わせていく
while h > 0 and w <= W:
    ans += len(list(comb([0] * (h+w-2), h-1))) * len(list(comb([0] * (H-h+W-w), H-h))) % (10 ** 9 + 7)
    h -= 1
    w += 1

print(ans % (10 ** 9 + 7))