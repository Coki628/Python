# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc025
　　　http://baitop.hatenadiary.jp/entry/2018/09/01/015743
　　　http://www.mwsoft.jp/programming/java/java_lang_integer_bit_count.html
　　　https://atcoder.jp/contests/abc025/submissions/5691594
　　　https://atcoder.jp/contests/abc025/submissions/6237513
・bitDP
・入れる数字を1から順番に進めていくことで、埋まっている場所さえbitで取っておけば
　入っている数字を考えなくて良くなる。(今やろうとしている数より小さいことが分かってるから)
・今、何番を入れようとしているかは、立っているビットを数えると分かる。
・演算子の優先順位、pythonだと^が==より強い。
・色々速度改善版。
・グリッドは1次元として考えると、ビットと紐付けてチェックが出来る。
・c++のほぼ同じコードは0.3秒。
・pythonもpypyもMLEだから、3000万のリストがやっぱ持てないんだろうね。。
・リストをnumpyのやつにしてメモリ節約。MLEはなくなったけどTLEは変わらず。。
"""

import sys
import numpy as np

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

grid=[None]*5
for i in range(5):
    grid[i]=LIST()

used=[None]*26
for i in range(5):
    for j in range(5):
        if grid[i][j]!=0:
            used[grid[i][j]]=(i, j)

def check(bit, a):
    h=a//5
    w=a%5
    # 縦位置の確認
    if 1<=h<=3:
        # グリッドでのaから見た上下
        u=(bit>>a-5)&1
        d=(bit>>a+5)&1
        # 上下どちらかだけ埋まってたらNG
        if u^d:
            return 0
    # 横位置の確認
    if 1<=w<=3:
        # グリッドでのaから見た左右
        l=(bit>>a-1)&1
        r=(bit>>a+1)&1
        # 左右どちらかだけ埋まってたらNG
        if l^r:
            return 0
    return 1

def bit_cnt(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

# dp[i] := 集合iを埋まっているマス目として、その状態でのうまくいく通り数
dp=np.zeros(1<<25, dtype=np.int32)
dp[0]=1
for bit in range((1<<25)-1):
    # 0からは何も遷移しない
    if dp[bit]==0: continue
    cur=bit_cnt(bit)+1
    # 置く場所が決まっている時はその遷移だけ確認
    if used[cur]:
        h,w=used[cur]
        nxt=h*5+w
        # nxtが既にbitに含まれている時はskip
        if (bit>>nxt)&1: continue
        dp[bit+(1<<nxt)]+=dp[bit]*check(bit, nxt)
        dp[bit+(1<<nxt)]%=MOD
    else:
        # 決まっていない時は25箇所それぞれを試す
        for i in range(25):
            # iが既にbitに含まれている時はskip
            if (bit>>i)&1: continue
            dp[bit+(1<<i)]+=dp[bit]*check(bit, i)
            dp[bit+(1<<i)]%=MOD
print(dp[-1])
