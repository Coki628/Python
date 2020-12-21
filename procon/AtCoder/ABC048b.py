# -*- coding: utf-8 -*-

a, b, x = map(int, input().split())

# これじゃ全然間に合わない…
# ans = 0
# for i in range(a, b+1):
#     if i % x == 0:
#         ans += 1
# print(ans)

all_num = b // x
lower_num = (a - 1) // x
print(all_num - lower_num)