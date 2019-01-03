# -*- coding: utf-8 -*-

from collections import deque

N, K = map(int, input().split())

if N < K:
    print('NO')
    exit()

que = deque()
# 大きい方から使って、まずは最低限の数で作る
for i in range(30, -1, -1):
    num = 2 ** i
    if N >= num:
        N -= num
        que.append(num)
# この時点で既に多すぎればだめ
if len(que) > K:
    print('NO')
    exit()
# 丁度いい数になるまで分解していく
while len(que) != K:
    cur = que.popleft()
    cur //= 2
    # 1はこれ以上崩せないから後ろに回す
    if cur != 1:
        que.appendleft(cur)
        que.appendleft(cur)
    else:
        que.append(cur)
        que.append(cur)

print('YES')
print(*(sorted(que)))
