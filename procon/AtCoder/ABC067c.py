# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

# 初期値設定
sunuke = aN[0]
arai = sum(aN) - aN[0]
ans = abs(sunuke - arai)

for i in range(1, N-1):
    sunuke += aN[i]
    arai -= aN[i]
    ans = min(ans, abs(sunuke - arai))
print(ans)