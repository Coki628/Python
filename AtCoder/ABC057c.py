# -*- coding: utf-8 -*-

N = int(input())

aN = []
i = 1
# かけ合わせる数の左右逆はやる意味ない
while i <= N//i:
    if N % i == 0:
        aN.append([i, N//i])
    i += 1

# 末尾に格納されたペアの右側の数の桁数
print(len(str(aN[-1][1])))