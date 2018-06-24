# -*- coding: utf-8 -*-

A, B, C, D = map(int, input().split())

# これだとfor回さなくて済む
ans = min(B, D) - max(A, C)
if ans > 0:
    print(ans)
else:
    print(0)