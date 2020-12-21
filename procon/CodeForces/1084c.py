# -*- coding: utf-8 -*-

"""
解説参考
・正規表現祭りにしてみた。
・+1してからかけて、最後に合計から-1ってのは自分からは出ないと思う…。
"""

import re

s = input()
mod = 10 ** 9 + 7

# アルファベットa,b以外を消す(^は[]の中で使うと先頭じゃなくて以外の意味)
s = re.sub(r'[^a-b]', '', s)
# 複数のbをひとつに
s = re.sub(r'b+', 'b', s)
# 先頭と末尾のbはいらない
s = re.sub(r'^b', '', s)
s = re.sub(r'b$', '', s)

s = s.split('b')
if s[0] != '':
    ans = 1
    for i in range(len(s)):
        ans = (ans * (len(s[i]) + 1)) % mod
    print(ans - 1)
else:
    print(0)
