# -*- coding: utf-8 -*-

import itertools

N = int(input())
name_list = []
for i in range(N):
    name_str = input()
    # 1文字目がMARCHの名前だけリストに詰める
    if name_str[0] == "M" or name_str[0] == "A" or name_str[0] == "R" or name_str[0] == "C" or name_str[0] == "H":
        name_list.append(name_str)
# 3人を選ぶ組み合わせ
cmb_list = list(itertools.combinations(name_list, 3))

ans = 0
for names in cmb_list:
    # 各人の名前のイニシャルを取得
    init1 = names[0][0]
    init2 = names[1][0]
    init3 = names[2][0]
    # 重複がなければ条件を満たす
    if init1 != init2 and init1 != init3 and init2 != init3:
        ans += 1
    
print(ans)