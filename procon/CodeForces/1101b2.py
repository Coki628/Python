# -*- coding: utf-8 -*-

"""
ハックされた…終わってる。
てことで条件1個追加
"""

s = input()

lb = s.find('[')
lc = lb+1 + s[lb+1:].find(':')
rb = s.rfind(']')
rc = s[:rb].rfind(':')

# [がない時、lbが-1でうまく並んでしまうので条件追加
if not lb < lc < rc < rb or lb == -1:
    print(-1)
    exit()

print(4 + s[lc:rc].count('|'))
