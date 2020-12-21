# -*- coding: utf-8 -*-

from itertools import product

N = int(input())

cnt = 0
# 候補となりうる9桁までの数字を列挙
for prod in product(['0', '3', '5', '7'], repeat=9):
    # チェック用に変形
    num = int(''.join(prod))
    s = str(num)
    # 753数ではない
    if (s.find('3') == -1 or s.find('5') == -1
            or s.find('7') == -1 or s.find('0') != -1):
        continue
    # N以下ではない
    if num > N:
        continue
    cnt += 1
print(cnt)
