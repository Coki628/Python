# -*- coding: utf-8 -*-

"""
つるかめ算をやってみる
"""

N, M = map(int, input().split())

for b in range(N+1):
    # 実際の足の数 - 全てaでcは0と仮定した時の足の総数
    all_a = M - (N-b) * 2 - b * 3 - 0 * 4
    # 4-2はaとcの足の本数の差分
    # 負数でないことと、小数にならない(割り切れる)こと
    if all_a / (4-2) >= 0 and all_a % (4-2) == 0:
        c = all_a // (4-2)
        # こちらも負数にならないこと
        if N-b-c >= 0:
            print(N-b-c, b, c)
            exit() 
print(-1, -1, -1)
