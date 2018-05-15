# -*- coding: utf-8 -*-

# 入力を各変数に格納
a, b = map(int, input().split())

ans = 0
for i in range(a, b+1):

    # 数値→文字列に
    str1 = str(i)
    # 1桁目と5桁目、2桁目と4桁目がそれぞれ同じ
    if str1[0] == str1[4] and str1[1] == str1[3]:
        ans += 1
    
    i += 1

print(ans)