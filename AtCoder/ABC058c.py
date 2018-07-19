# -*- coding: utf-8 -*-

n = int(input())
Sn = [input() for i in range(n)]

# 各文字列が含むアルファベットを集計
az_list = []
for i in range(n):
    s_list = list(Sn[i])
    az_dict = {}
    for c in s_list:
        if c not in az_dict:
            az_dict[c] = 1
        else:
            az_dict[c] += 1
    az_list.append(az_dict)

# 集計結果から、全ての文字列が共通で持っているアルファベットとその数を出す
sum_dict = {}
for i in range(97, 97+26):
    c = chr(i)
    for j in range(len(az_list)):  
        if c not in az_list[j]:
            sum_dict[c] = 0
            break
        else:
            if j == 0:
                sum_dict[c] = az_list[j][c]
            else:
                sum_dict[c] = min(sum_dict[c], az_list[j][c])

# 辞書順で解答となる文字列を作っていく
ans = ""
for i in range(97, 97+26):
    c = chr(i)
    if c in sum_dict:
        for j in range(sum_dict[c]):
            ans += c

print(ans)