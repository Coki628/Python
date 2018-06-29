# -*- coding: utf-8 -*-

N = int(input())

ans = 1
for i in range(N):
    # 調整の処理は都度行っても端数の側に影響を与えないので問題ない
    ans = (ans * (i + 1)) % (10 ** 9 + 7) 

# 大き過ぎる値を調整する
# ans = ans % (10 ** 9 + 7) 

print(ans)