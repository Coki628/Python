# -*- coding: utf-8 -*-

N, M, Q = map(int, input().split())
lM, rM = [0] * M, [0] * M
pQ, qQ = [0] * Q, [0] * Q
# 都市番号を0-indexed
for i in range(M):
    l, r = map(int, input().split())
    lM[i], rM[i] = l-1, r-1
for i in range(Q):
    p, q = map(int, input().split())
    pQ[i], qQ[i] = p-1, q-1

# (始点, 終点)
cities = [(set(), set()) for i in range(N)]
for i in range(M):
    cities[lM[i]][0].add(i)
    cities[rM[i]][1].add(i)

# 各質問
for i in range(Q):
    starts = set()
    ends = set()
    # 各都市を始点終点としている列車番号を集めていく
    for j in range(pQ[i], qQ[i]+1):
        starts |= cities[j][0]
        ends |= cities[j][1]
    print(len(starts & ends))
