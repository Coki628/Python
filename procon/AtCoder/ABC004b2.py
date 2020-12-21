# -*- coding: utf-8 -*-

"""
真面目に自分で実装した版
"""

field = []
for i in range(4):
    field.append(list(input()))
# 縦も横も逆順にして出力
for i in range(3, -1, -1):
    field[i].reverse()
    print(''.join(field[i]))
