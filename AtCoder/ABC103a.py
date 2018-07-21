# -*- coding: utf-8 -*-

aN = list(map(int, input().split()))

aN.sort(reverse=True)

ans = 0
for i in range(1,len(aN)):
    ans += aN[i-1] - aN[i]
print(ans)