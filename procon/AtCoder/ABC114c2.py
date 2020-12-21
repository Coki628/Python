# -*- coding: utf-8 -*-

"""
解説参考
再帰関数で候補の数を作っていく版
"""

N = int(input())

def dfs(s):
    # N以下しかやらない
    if int(s) > N:
        return 0
    # 753数かどうか
    if '7' in s and '5' in s and '3' in s:
        res = 1
    else:
        res = 0
    # sの後ろに753を付けたもので再帰していく
    for c in '753':
        res += dfs(s + c)
    # 最終的に自分より大きい桁で増えた分を返却していく
    return res

print(dfs('0'))
