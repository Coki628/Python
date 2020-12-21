# -*- coding: utf-8 -*-

S = input()

for i in range(1, len(S)):
    # 必要な部分の切り出し
    str1 = S[:len(S)-i]
    # 文字数が奇数ならそもそも論外
    if len(str1) % 2 != 0:
        continue
    # 前半部分と後半部分の比較
    tmp = len(str1)/2
    if str1[:len(str1)//2] == str1[len(str1)//2:]:
        print(len(str1))
        exit()