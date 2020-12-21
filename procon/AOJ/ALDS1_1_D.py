# -*- coding: utf-8 -*-

N = int(input())
aN = [int(input()) for i in range(N)]

start = aN[0]
ans = -float('inf')
for i in range(1,N):
    # 今-起点と今までの最大を比較
    ans = max(aN[i] - start, ans)
    # 最安値の更新したら起点を変える
    start = min(aN[i], start)
print(ans)