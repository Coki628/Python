# -*- coding: utf-8 -*-

a, b = map(int, input().split())

# 計算で全部出せるとfor文回さなくて済む
ans = (b - a - 1) * (b - a) // 2 - a
print(ans)