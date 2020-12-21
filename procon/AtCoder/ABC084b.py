# -*- coding: utf-8 -*-

A, B = map(int, input().split())
S = input()

# 部分文字列を取得して、各条件に合致するか確認
if S[:A].isdigit() and S[A] == "-" and S[A+1:A+1+B].isdigit():
    print("Yes")
else:
    print("No")