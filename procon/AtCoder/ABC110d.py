# -*- coding: utf-8 -*-

"""
とりあえず自力で
→盛大にTLE
"""

from math import sqrt
from itertools import permutations, combinations_with_replacement
import numpy as np

def num_div_set2(N):
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て+1
    end = int(sqrt(N)) + 1
    for i in range(2, end+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N, M = map(int, input().split())
MOD = 10 ** 9 + 7

ans = 0
# 約数で重複組み合わせ
for comb in combinations_with_replacement(num_div_set2(M), N):
    # 配列要素全部かける
    if np.prod(comb) == M:
        # 順列から重複削除
        ans = (ans + len(set(permutations(comb)))) % MOD
print(ans)
