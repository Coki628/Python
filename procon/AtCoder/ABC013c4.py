# -*- coding: utf-8 -*-

"""
計算で解く2(解説参考)
式変形してjについて解けば内ループなくせる
"""

N, H = map(int, input().split())
A, B, C, D, E = map(int, input().split())

ans = float('inf')
for i in range(N+1):
    # jについて式変形
    j = ((N-i) * E - H - B*i) / (D + E)
    # 0未満は0にする
    if j < 0:
        j = 0
    # >=じゃなくて>なので、常に大きくするために、切り捨て+1
    else:
        j = int(j) + 1
    ans = min(A*i + C*j, ans)
print(ans)
