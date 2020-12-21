# -*- coding: utf-8 -*-

N = int(input())

# 割り切れるかどうかで見ると綺麗にまとまる
if N % 1111:
    print('DIFFERENT')
else:
    print('SAME')