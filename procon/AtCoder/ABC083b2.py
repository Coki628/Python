# -*- coding: utf-8 -*-

N, A, B = map(int, input().split())

ans = 0
# 1～Nのループ
for i in range(1, N+1):

    sum1 = 0
    num = i
    # 各桁の値を合計していく
    while num > 0:
        sum1 += num % 10
        # 整数を返させる書き方
        num //= 10
    # 条件に合っていれば答えに足す
    if sum1 >= A and sum1 <= B:
        ans += i

print(ans)