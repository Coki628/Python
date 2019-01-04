# -*- coding: utf-8 -*-

"""
めっちゃ時間かかったけど自力AC！
"""

N, K = map(int, input().split())

# 余り0以上は全ての組
if K == 0:
    print(N*N)
    exit()

ans = 0
# 割られる数でループ
for i in range(1, N+1):
    # 割られる数がK以下ならmodしてKはあり得ない
    if i <= K:
        continue
    # 確実に入る分
    ans += N // i * (i-K)
    # 最後の分を調整
    ans += max(N % i - K + 1, 0)
print(ans)
