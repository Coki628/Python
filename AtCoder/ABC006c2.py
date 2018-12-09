# -*- coding: utf-8 -*-

N, M = map(int, input().split())

# いける範囲内かどうか
if N * 2 <= M <= N * 4:
    # 偶奇で分ける
    if M % 2 == 0:
        for i in range(N+1):
            # 偶数は2と4のみで組み合わせる
            if i * 2 + (N-i) * 4 == M:
                print(i, 0, N-i)
                exit()
    else:
        for i in range(N):
            # 奇数は3を1回固定で入れてあと2と4で合わせる
            if i * 2 + (N-i-1) * 4 == M - 3:
                print(i, 1, N-i-1)
                exit()
print(-1, -1, -1)
