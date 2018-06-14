# -*- coding: utf-8 -*-

import itertools

N = int(input())
name_dict = {"M":0, "A":0, "R":0, "C":0, "H":0}

# 各イニシャル毎に集計
for i in range(N):
    name_str = input()
    if name_str[0] == "M":
        name_dict["M"] += 1
    if name_str[0] == "A":
        name_dict["A"] += 1
    if name_str[0] == "R":
        name_dict["R"] += 1
    if name_str[0] == "C":
        name_dict["C"] += 1
    if name_str[0] == "H":
        name_dict["H"] += 1

# MARCHから3文字を選ぶ組み合わせ
cmb_list = list(itertools.combinations(["M","A","R","C","H"], 3))

ans = 0
for cmb in cmb_list:
    # 各イニシャルの人数毎に掛け合わせて合計していく
    ans += name_dict[cmb[0]] * name_dict[cmb[1]] * name_dict[cmb[2]]
    
print(ans)