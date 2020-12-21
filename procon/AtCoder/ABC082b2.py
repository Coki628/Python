# -*- coding: utf-8 -*-

s_list = list(input())
t_list = list(input())

# 昇順ソート
s_list.sort()
# 降順ソート
t_list.sort(reverse=True)

# 文字列の並び順も比較演算子で大小比較できる
if "".join(s_list) < "".join(t_list):
    print("Yes")
else:
    print("No")