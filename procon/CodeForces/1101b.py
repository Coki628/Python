# -*- coding: utf-8 -*-

s = input()

lb = s.find('[')
lc = lb+1 + s[lb+1:].find(':')
rb = s.rfind(']')
rc = s[:rb].rfind(':')

if not lb < lc < rc < rb:
    print(-1)
    exit()

print(4 + s[lc:rc].count('|'))
