# -*- coding: utf-8 -*-

import itertools

A, B, C, D = map(str, list(input()))

# 重複順列でオペランド組み合わせの全パターンを出す
op_list = list(itertools.product(["+", "-"], repeat=3))

for op in op_list:
    # evalで文字列を式にできる(これが出なかった…)
    if eval(A + op[0] + B + op[1] + C + op[2] + D) == 7:
        print(A + op[0] + B + op[1] + C + op[2] + D + "=7")
        exit()