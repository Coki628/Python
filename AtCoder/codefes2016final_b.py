# -*- coding: utf-8 -*-

"""
参考：http://www.geisya.or.jp/~mwm48961/kou2/cos_rule.htm
　　　https://mathtrain.jp/heron
　　　https://mathtrain.jp/naisetsuen
・数学系
・参考は上から、余弦定理、ヘロン、内接円
・まあ、結局解けなかったんだけどね。
"""

import sys
from math import hypot, sqrt

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

x1,y1=MAP()
x2,y2=MAP()
x3,y3=MAP()

# 各辺の長さ
a=hypot(x1-x2, y1-y2)
b=hypot(x2-x3, y2-y3)
c=hypot(x3-x1, y3-y1)
# 三角形の面積(ヘロンの公式の変形)
S=1/4*sqrt(2*(a**2*b**2+b**2*c**2+c**2*a**2)-(a**4+b**4+c**4))
# 3辺の長さから角ABC(余弦定理の変形)
cosA=(b**2+c**2-a**2)/(2*b*c)
cosB=(c**2+a**2-b**2)/(2*c*a)
cosC=(a**2+b**2-c**2)/(2*a*b)

# 元の三角形を半分に切る
s=S/2
if max(a, b, c)==a:
    # 辺aの長さを半分にした時の辺c(余弦定理)
    c2=sqrt((a/2)**2+b**2-2*(a/2)*b*cosC)
    # c2とa/2を使って、内接円の半径を出す
    print(2*s/(a/2+b+c2))
elif max(a, b, c)==b:
    a2=sqrt((b/2)**2+c**2-2*(b/2)*c*cosA)
    print(2*s/(a2+b/2+c))
else:
    b2=sqrt((c/2)**2+a**2-2*(c/2)*a*cosB)
    print(2*s/(a+b2+c/2))
    # 切り出した反対側の三角形の内接円。ちょっとサイズ違う。
    print(2*s/(b+b2+c/2))
    # 真ん中と取ってもサンプルと合わない。打ち止め。
    print((2*s/(a+b2+c/2)+2*s/(b+b2+c/2))/2)
