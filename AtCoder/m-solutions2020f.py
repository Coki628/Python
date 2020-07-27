"""
参考：https://img.atcoder.jp/m-solutions2020/editorial.pdf
・自力ならず。
・座標系、回転
・180度でぶつかるのはさておき、90度でぶつかるパターンについて重要な考察、
　x+yが一致するものだけが衝突の可能性がある、という話。
・最初聞いても意味がよく分からず、「(-1,-1)と(2,2)とかx+y全然一致しないけどぶつかるかもじゃん、、」
　って思ったんだけど、よくよく読み込んでやっと理解。これはUとRに限った話なんだよね。。
　UとRだとどっちも時間が経つ毎にx+yの値は+1されていって一致している状況は変わらず、
　それがいつか同じ座標になったり、方向によってはならなかったりする。
・で、これなら話が簡単になるから、これを他にも適用できるように、90度ずつ座標を回転させていって、
　それぞれをUとRみたいな関係の位置にしてやる。これを4回やればOK。
・最初TLEして、内ループでペア対象全部見てるのがダメだったので、にぶたんで必要なところだけ見るようにしてAC。
　(Rのy座標がちょうどUのy座標より大きくなるところだけ分かればいい。)
・あと一応180度の方、こっちのが簡単だけど、座標毎に集めてソートして、尺取の弱体化版みたいなやつは
　やらないといけないから、地味に思ったよりはちょっとめんどかった。
"""

import sys
from collections import defaultdict
from bisect import bisect_right

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
XYD = []
for _ in range(N):
    x, y, d = input().split()
    x = int(x)
    y = int(y)
    if d == 'U':
        XYD.append([x, y, 0])
    elif d == 'R':
        XYD.append([x, y, 1])
    elif d == 'D':
        XYD.append([x, y, 2])
    else:
        XYD.append([x, y, 3])

# 180度の方の準備
DX = defaultdict(list)
DY = defaultdict(list)
for x, y, d in XYD:
    if d == 0:
        DX[x].append((y, 0))
    elif d == 1:
        DY[y].append((x, 0))
    elif d == 2:
        DX[x].append((y, 1))
    else:
        DY[y].append((x, 1))

ans = INF
for li in DX.values():
    li.sort()
    mxl = -1
    for y, d in li:
        # Uが来たら左端とする
        if d == 0:
            mxl = y
        # Dが来たらそこを右端として一番右にある左端との距離を取る
        elif d == 1 and mxl != -1:
            ans = min(ans, (y-mxl) * 10 // 2)
for li in DY.values():
    li.sort()
    mxl = -1
    for x, d in li:
        if d == 0:
            mxl = x
        elif d == 1 and mxl != -1:
            ans = min(ans, (x-mxl) * 10 // 2)

for d1 in range(4):
    d2 = (d1+1) % 4
    D = defaultdict(list)
    # まずはUの方をx+y毎に隣接リストに詰める
    for x, y, d in XYD:
        if d == d1:
            D[x+y].append(y)
    # この後のにぶたん用に番兵とソート
    for k in D.keys():
        D[k].append(-INF)
        D[k].sort()
    # Uが整ったのでRを見る
    for x, y, d in XYD:
        if d == d2:
            xy = x + y
            if xy in D:
                # Rのy座標がちょうどUのy座標より大きくなるところ
                idx = bisect_right(D[xy], y) - 1
                if D[xy][idx] != -INF:
                    y2 = D[xy][idx]
                    x2 = xy - y2
                    ans = min(ans, (abs(x-x2)+abs(y-y2)) * 10 // 2)
    # 90度回転の処理、偶奇毎にx,yの符号を変える
    if d1 % 2 == 0:
        for i in range(N):
            XYD[i][0] *= -1
    else:
        for i in range(N):
            XYD[i][1] *= -1
if ans == INF:
    print('SAFE')
else:
    print(ans)
