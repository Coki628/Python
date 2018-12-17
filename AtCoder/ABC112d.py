# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/11/06/120600
　　　https://emtubasa.hateblo.jp/entry/2018/10/06/224022
・約数とか。理解は出来たものの、自分で気付けるにはどうすれば…。
・この前作ったけど使えなかった約数列挙ライブラリが役に立った。
"""

from math import sqrt

def num_div_set2(N):
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て+1
    end = int(sqrt(N)) + 1
    for i in range(2, end+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N, M = map(int, input().split())

ans = 1
for d in num_div_set2(M):
    # aNを全てdとするより大きいのは、Mに収まらないからダメ
    # 逆にそれ以下の約数なら、全dの最後の1つだけ調整すればどうにでも作れるからOK
    if d * N <= M:
        ans = max(d, ans)
print(ans)
