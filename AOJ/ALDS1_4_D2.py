# -*- coding: utf-8 -*-

"""
二分探索の応用
"""

N, K = map(int, input().split())
aN = [int(input()) for i in range(N)]

# 同じ荷物を2回積むことはないので、2重ループでも計算量はO(N)
def check(P):
    # 積んだ荷物の個数
    load = 0
    # トラックの台数ループ
    for i in range(K):
        truck = 0
        # あるトラックに荷物を積み込むループ
        while truck + aN[load] <= P:
            truck += aN[load]
            load += 1
            # 全部積めた
            if load == N:
                return N
    # i個詰めた
    return load

# 2分探索で最初にN個全部積める場所を見つける
hi = 10 ** 9
low = 0
# low+1 == hi -> lowとhiが隣り合った状態なので、境界が確定する
while low+1 < hi:
    mid = (hi+low) // 2
    if check(mid) == N:
        hi = mid
    else:
        low = mid
# hiの位置が最初にN個全部積める積載量S
print(hi)
