# -*- coding: utf-8 -*-

"""
ちゃんとビット演算やる版
"""

N, X = map(int, input().split())
aN = list(map(int, input().split()))

ans = 0
for i in range(N):
    # Xをiビット右に動かした時、1桁目が1かどうか(1 & 1 == 1にできるか)
    if X >> i & 1:
        ans += aN[i]
print(ans)
