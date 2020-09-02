"""
・なんとか自力AC。危なかった。。本番なら絶対間に合ってない。。
・式変形、前計算、集計
・決める変数は4つあるけど、3つ決めれば残り1つは定まる。これはよくある。
　がしかしこれだとN^4→N^3になるまでなので足りない。
・C++でなんとか行けないか試した。→TLE。。
・最大と最小だけ取っておいてr-lで区間内をまとめて数えあげようとした。
　→区間内の値に全ての整数値で解がある訳じゃないのでダメ。。
・なんとか答えがある場所だけに注目したい。。ん？？
　ここで、2つ分だけ前計算して、後から残りをがっちゃんこ、を思い立つ。
・無事これでAC1.77秒。Counterの集計が400万もあるとキツいらしく、
　最初TLEで、後半パートで値のあり得ないとこ触らないように条件付けたらAC。
・ちなみに公式によると、これは半分全列挙の1種になるそうな。
　指数の制約が絡まない半分全列挙とか珍しいなー、なんて思ったり。
・ちょっと調整して、Counter使わないで配列にした版。速い。AC0.23秒。
　やっぱ物量多い時はなるべくdictじゃなくてlist使うべきだな。
"""

import sys

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

N, D = MAP()

NN = N**2
C = [0] * (NN*2+1) 
for x in range(1, N+1):
    xx = x**2
    for y in range(1, N+1):
        yy = y**2
        C[xx+yy] += 1

ans = 0
for w in range(1, N+1):
    ww = w**2
    for z in range(1, N+1):
        zz = z**2
        xxyy = ww + D - zz
        if xxyy > NN*2:
            continue
        if xxyy < 0:
            break
        ans += C[xxyy]
print(ans)
