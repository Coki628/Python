# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc072/editorial.pdf
　　　http://potetisensei.hatenablog.com/entry/2018/03/23/231710
　　　http://ronly.hatenablog.com/entry/2017/10/12/182048
・後手が勝てるケースが限定的なのは何となく思ったんだけど…。
・こういうのを思いつくのは大変。
・結果から遡ってうまく規則性が割り出せるようになればいいんだが…。
・そのうちGrundy数とか調べてみるか。
"""

X, Y = map(int, input().split())

if abs(X - Y) <= 1:
    print('Brown')
else:
    print('Alice')
