# -*- coding: utf-8 -*-

C,D = map(int, input().split())

ans = 0
i = 0
while 140 * (2**i) <= 10 ** 15:

    ans += max(min(170 * (2**i), D) - max(140 * (2**i), C), 0)
    i += 1

print(ans)