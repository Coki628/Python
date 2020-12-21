# -*- coding: utf-8 -*-

"""
参考：https://nuekodory.github.io/2018/06/24/AtCoder-ABC101-ARC-099-%E3%82%92-Python3-%E3%81%A7%E8%A7%A3%E3%81%8F/
・調査用。
・完全に未知の物が現れた時はこうやって値の動きを確認するのも良さそう。
"""

import matplotlib.pyplot as plt

def make_snk(num):
    s = str(num)
    res = 0
    for c in s:
        res += int(c)
    return res

snk_list = [0]
for i in range(1, 100000):
    snk_list.append(i / make_snk(i))

plt.plot(snk_list[:1000])
plt.ylabel('n / S(n)')
plt.xlabel('n')
plt.show()

plt.plot(snk_list[1000:2000])
plt.ylabel('n / S(n)')
plt.xlabel('n')
plt.show()

plt.plot(snk_list[10000:])
plt.ylabel('n / S(n)')
plt.xlabel('n')
plt.show()
