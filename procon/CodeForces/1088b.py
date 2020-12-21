# -*- coding: utf-8 -*-

N, K = map(int, input().split())
aN = list(map(int, input().split()))

aN = sorted(set(aN))
# aNのが短い分調整
aN.extend([0] * (K - len(aN)))

print(aN[0])
for i in range(1, K):
    if aN[i] != 0:
        # ひとつ前との差分を出力してく
        print(aN[i] - aN[i-1])
    else:
        print(0)
