# -*- coding: utf-8 -*-

t = int(input())

for i in range(t):
    n, x, y, d = map(int, input().split())
    cnts = []
    # なるべく近づく
    if abs(x - y) % d == 0:
        cnts.append(abs(x - y) // d)
    else:
        cnts.append(float('inf'))
    # いったん1ページ目に来て進む
    if (y-1) % d == 0:
        # x-1 / d の切り上げ => xから1ページ目
        # y-1 // d          => 1ページ目からy
        cnts.append(-(-(x-1) // d) + (y-1) // d)
    else:
        cnts.append(float('inf'))
    # いったん最終ページに来て戻る
    if (n-y) % d == 0:
        # n-x / d の切り上げ => xから最終ページ
        # n-y // d          => 最終ページからy
        cnts.append(-(-(n-x) // d) + (n-y) // d)
    else:
        cnts.append(float('inf'))
    # 3パターンのうち最小の値
    if min(cnts) == float('inf'):
        print(-1)
    else:
        print(min(cnts))
