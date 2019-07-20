# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc025
　　　http://baitop.hatenadiary.jp/entry/2018/09/01/015743
　　　http://www.mwsoft.jp/programming/java/java_lang_integer_bit_count.html
・bitDP
・入れる数字を1から順番に進めていくことで、埋まっている場所さえbitで取っておけば
　入っている数字を考えなくて良くなる。(今やろうとしている数より小さいことが分かってるから)
・今、何番を入れようとしているかは、立っているビットを数えると分かる。
・演算子の優先順位、pythonだと^が==より強い。
・計算量はbitループの2^25と内ループのマス目分25とcheck関数内の25で
　2^25*25^2=3300万*625...625はcontinueもあるし全部じゃないけど、
　これpythonで通る訳ないよね。。(pythonは試してもないけどpypyはMLEだった)
・c++で書いたほぼ同じコードでAC、それでも3.5秒くらい。(この問題、制限5秒)
"""

import sys

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

grid2=list2d(7, 7, -1)
def check(bit, a):
    cnt=0
    for i in range(1, 6):
        for j in range(1, 6):
            if cnt==a:
                h,w=i,j
            grid2[i][j]=(bit>>cnt)&1
            cnt+=1
    # 縦方向のチェック
    if grid2[h-1][w]!=-1 and grid2[h+1][w]!=-1 \
            and (grid2[h-1][w]==1) ^ (grid2[h+1][w]==1):
        return 0
    # 横方向のチェック
    if grid2[h][w-1]!=-1 and grid2[h][w+1]!=-1 \
            and (grid2[h][w-1]==1) ^ (grid2[h][w+1]==1):
        return 0
    return 1
    
# def bit_cnt(bit):
#     res=0
#     for i in range(25):
#         res+=(bit>>i)&1
#     return res
def bit_cnt(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

# dp[i] := 集合iを埋まっているマス目として、その状態でのうまくいく通り数
dp=[0]*(1<<25)
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
        dp[bit+2**nxt]+=dp[bit]*check(bit, nxt)
        dp[bit+2**nxt]%=MOD
    else:
        # 決まっていない時は25箇所それぞれを試す
        for i in range(25):
            # iが既にbitに含まれている時はskip
            if (bit>>i)&1: continue
            dp[bit+2**i]+=dp[bit]*check(bit, i)
            dp[bit+2**i]%=MOD
print(dp[-1])
