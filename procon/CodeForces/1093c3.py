# -*- coding: utf-8 -*-

"""
人の参考
あとifの条件ひとつ足りなかった…。
"""

N = int(input())
bN = list(map(int, input().split()))

aN = [0] * N
aN[0] = 0
aN[-1] = bN[0]
for i in range(1, N // 2):
    # (今回のb - 前回の前a)を、前回の後ろaと比べる
    # ※前回の前aの分は確実に前に必要なため引く
    if bN[i] - aN[i-1] <= aN[N-i]:
        # 前を最大限小さく
        aN[i] = aN[i-1]
        # 後ろに残り
        aN[N-i-1] = bN[i] - aN[i-1]
    else:
        # 前に残り
        aN[i] = bN[i] - aN[N-i]
        # 後ろを最大限大きく
        aN[N-i-1] = aN[N-i]
print(*aN)
