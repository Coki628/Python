# -*- coding: utf-8 -*-

""" 逆順使わないでDPでやる版 """

S = input()

dp = [False] * (len(S)+1)
# 0文字地点は遡る用にTrueにしておく
dp[0] = True

for i in range(1, len(dp)):
    if i-5 >= 0:
        if S[i-5:i] == "dream" or S[i-5:i] == "erase":
            # 条件に当てはまったら、文字数分遡って有効な文字列が続いているか確認
            if dp[i-5]:
                dp[i] = True
    if i-6 >= 0:
        if S[i-6:i] == "eraser":
            if dp[i-6]:
                dp[i] = True
    if i-7 >= 0:
        if S[i-7:i] == "dreamer":
            if dp[i-7]:
                dp[i] = True

if dp[len(S)]:
    print("YES")
else:
    print("NO")