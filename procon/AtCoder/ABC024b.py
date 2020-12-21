# -*- coding: utf-8 -*-

N,T = map(int, input().split())

# 最後の人が通ってから開き続ける時間も見る
time = 10 ** 6 + 1 + 10 ** 5
people = [False] * time

# 全時間のうち人が通る瞬間を保持
for i in range(N):
    p = int(input())
    people[p] = True

cnt,ans = 0,0
# 全時間ループ
for i in range(time):
    # 人が通った
    if people[i]:
        cnt = T
    # 空いてるか確認
    if cnt != 0:
        ans += 1
    # 空いてる時間減らす
    cnt = max(cnt-1, 0)

print(ans)