# -*- coding: utf-8 -*-

s_list = list(input())
t_list = list(input())

# 昇順ソート
s_list.sort()
# 降順ソート
t_list.sort(reverse=True)

res_list = []
res_list.append("".join(s_list))
res_list.append("".join(t_list))
res_list.sort()

# 比較対象が同一文字列の場合はNoにする(s<=tではなくs<tだから)
if "".join(s_list) == "".join(t_list):
    print("No")
# ソート後の先頭と文字列sの一致を確認
elif res_list[0] == "".join(s_list):
    print("Yes")
else:
    print("No")