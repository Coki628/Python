# -*- coding: utf-8 -*-

num1 = 2.65

print(round(num1, 1))

from decimal import Decimal, ROUND_HALF_UP

b = Decimal(str(num1))
c = b.quantize(Decimal(".1"), ROUND_HALF_UP)
print(float(c))