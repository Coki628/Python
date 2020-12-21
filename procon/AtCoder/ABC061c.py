# -*- coding: utf-8 -*-

N, K = map(int, input().split())

N_dict = {}
for i in range(N):
    a, b = map(int, input().split())
    if a not in N_dict:
        N_dict[a] = b
    else:
        N_dict[a] += b

# dictをkeyでソート
N_sorted = sorted(N_dict.items(), key=lambda x: x[0])

cnt = K
for n in N_sorted:
    cnt -= n[1]
    if cnt <= 0:
        print(n[0])
        exit()