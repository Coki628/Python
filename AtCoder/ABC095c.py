# -*- coding: utf-8 -*-

a, b, ab, x, y = map(int, input().split())

sum_cost = 0

# 両方必要な部分の計算
if x <= y:
    ryoho = x
else:
    ryoho = y
if (a + b)/ 2 <= ab:
    sum_cost += (a * ryoho)
    sum_cost += (b * ryoho)
else:
    sum_cost += (ab * ryoho * 2)

# 片方が多く必要な部分の計算
kataho = abs(x - y)
if x >= y:
    if a <= ab * 2:
        sum_cost += (a * kataho)
    else:
        sum_cost += (ab * kataho * 2)
else:
    if b <= ab * 2:
        sum_cost += (b * kataho)
    else:
        sum_cost += (ab * kataho * 2)

print(sum_cost)