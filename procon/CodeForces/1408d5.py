"""
参考：https://codeforces.com/blog/entry/83233
　　　他、色々な人たちのツイートとかコード
・長かった。。ここまで来るのに色々回り道してきた。
・まず基本方針は、acc[c] := 横位置cにあるサーチライトの縦位置d みたいなやつを作って、
　横位置を前から順に見ていくとすると、縦位置はc+1以降の最大値をチェックすればいいことできる。
・で、この時に全てのロバーについて見て、各ロバーの位置を加味した値にして、
　全ロバーでの最大を取れば、「横にx動かした時に最低限動かさないといけない縦y」を出せる。
・これを全てのxについて見れば、x+yの最小値が答え。
　ただし、愚直に全部のxを見るのは、100万*2000*logくらいの計算量になるので全然無理。(これが最初にやつ)
　ここから色々高速化に入る。
・まずxの値に対してyの変化する回数は、せいぜいサーチライトの数くらいしかないんじゃないかと思う。
　であればyが変化しない区間について全部チェックするのは無駄なので、にぶたんで前に進んでみる。(これが2番目)
・計算量的には、100万だったところが2000*logになってると思うので、
　なんとかなるんじゃないかと思ったけどTLE。。。
・それからセグ木をSparse Tableにしてみたり(3番目)、
　そもそも端からの区間maxしか取らないので累積maxにしたり(4番目)、
　C++で書いてみたりしたけどダメ。。
・ここで何人かの人のコードを眺めたりして、どんなことしてるか見てみる。
　あと、公式解説から、ロバーとサーチライトの全ペアについて何かチェックするようなことを言われる。
・すると、全てのロバーについてy = acc[a+x]-b+1ってやってるところで、
　これを累積maxの前計算に組み込めるんじゃあ、ってなる。
　で、累積max配列に対してacc[c] = dってしてたところを、全ペアについてchmax(acc[c-a], d-b+1)
　みたいに予めロバーのa,bを加味した値で最大取ってみることにする。
　これなら全てのxについて見る100万のループの内側はy = acc[x]だけで済むはず。
・最終的にこれで無事AC。まーじで長かった。疲れたけどちゃんと解けてよかった。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

N, M = MAP()
AB = []
for i in range(N):
    a, b = MAP()
    AB.append((a, b))
CD = []
for i in range(M):
    c, d = MAP()
    CD.append((c, d))

C, D = zip(*CD)
L = max(C)
acc = [0] * (L+2)
for a, b in AB:
    for c, d in CD:
        if c-a >= 0:
            acc[c-a] = max(acc[c-a], d-b+1)
acc = list(accumulate(acc[::-1], max))[::-1]

ans = INF
for x in range(L+2):
    y = acc[x]
    ans = min(ans, x+y)
print(ans)
