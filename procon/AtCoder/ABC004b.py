# -*- coding: utf-8 -*-

import numpy as np

field = []
for i in range(4):
    field.append(list(input()))

field = np.array(field)
# k=2で90度回転2回
field = np.rot90(field, k=2)

for i in range(4):
    print(''.join(field[i]))
