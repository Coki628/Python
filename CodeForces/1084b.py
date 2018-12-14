# -*- coding: utf-8 -*-

N, S = map(int, input().split())
aN = list(map(int, input().split()))

if (sum(aN) - S) / N < 0:
    print(-1)
else:
    print((sum(aN) - S) // N)
