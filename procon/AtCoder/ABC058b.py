# -*- coding: utf-8 -*-

O = input()
E = input()

S = ""
for i in range(len(E)):
    S += O[i]
    S += E[i]
# 必要に応じて最後の文字を追加
if len(O) > len(E):
    S += O[-1]

print(S)