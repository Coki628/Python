# -*- coding: utf-8 -*-

cnt = int(input())
card_list = [int(i) for i in input().split()]

# 2人の得点
alice = 0
bob = 0

# 高得点順にカードを並べ替え
card_list.sort(reverse=True)

for i in range(cnt):
    # 0開始で偶数ならAlice、奇数ならBobに加点する
    if i % 2 == 0:
        alice += card_list[i]
    else:
        bob += card_list[i]
# 得点差を出力
print(alice - bob)