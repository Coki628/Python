# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc069/editorial.pdf
　　　https://atcoder.jp/contests/abc055/submissions/4021548
・4通りのループをちょっと参考にしてすっきりさせた。
"""

from itertools import product

N = int(input())
s = list(input())

def verify(t):
    # 2か所決まれば芋づる式に全て定まる
    for i in range(1, N-1):
        # 羊がoと言ってるor狼がxと言ってる
        if t[i] == 'S' and s[i] == 'o' or t[i] == 'W' and s[i] == 'x':
            # 両隣は同じ
            if t[i-1] == 'S':
                t[i+1] = 'S'
            else:
                t[i+1] = 'W'
        # 逆
        else:
            # 両隣は違う
            if t[i-1] == 'S':
                t[i+1] = 'W'
            else:
                t[i+1] = 'S'
    # 終点手前 - 終点 - 始点、の辻褄が合ってるか検証する
    if t[-1] == 'S' and s[-1] == 'o' or t[-1] == 'W' and s[-1] == 'x':
        if not (t[-2] == 'S' and t[0] == 'S' or t[-2] == 'W' and t[0] == 'W'):
            return False
    else:
        if not (t[-2] == 'S' and t[0] == 'W' or t[-2] == 'W' and t[0] == 'S'):
            return False
    # 終点 - 始点 - 始点の次、の辻褄が合ってるか検証する
    if t[0] == 'S' and s[0] == 'o' or t[0] == 'W' and s[0] == 'x':
        if not (t[-1] == 'S' and t[1] == 'S' or t[-1] == 'W' and t[1] == 'W'):
            return False
    else:
        if not (t[-1] == 'S' and t[1] == 'W' or t[-1] == 'W' and t[1] == 'S'):
            return False
    return True

# 2か所決める方法4通り
for a, b in product('SW', repeat=2):
    t = [''] * N
    t[0] = a
    t[1] = b
    if verify(t): 
        print(''.join(t))
        exit()
print(-1)
