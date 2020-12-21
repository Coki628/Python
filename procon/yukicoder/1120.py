"""
・なーんとか、自力AC。。1回寝かせたりして本当に色々考えた末ね。。
・考察頑張る。場合分け。
・操作回数が一意なことや、mod (N-2)で0じゃないと無理なことなど、ある程度の手がかりは得ていたものの、
　なかなかそこから先に進めず。どうにもいろんなケースがありそうで考えがまとまらない。
・風呂で色々考えてて、1つ核心に迫る事実に気付く。
　一度に加点できるのは1人で他は全員減点されるので、加点が欲しい人が2人以上いると、
　その2人だけでやったとしても打ち消し合って少しも増えない、つまり加点が必要な人は常に1人に限られる。
　これを使えればだいぶあり得るケースが絞れそう、となる。
・まあそれでもこの後もだいぶガチャガチャ試行錯誤した。偶奇で無理なケースを潰したり。
　で、明らかにたくさん減らしたいやつが1人いたら無理だろ、ってなって、
　最大限減らしたい人が、残ってる操作回数より多く減らさないといけない時は無理、
　の条件足して、やーっと通った。
・あと色々詳細はソース内コメントにて。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()
A = LIST()
B = LIST()

# とりあえずA-Bして全員0を目指す形にする
AB = [0] * N
for i in range(N):
    AB[i] = A[i] - B[i]

# N=2は0除算になるので別処理
ans = 0
if N == 2:
    if AB[0] < AB[1]:
        AB[0], AB[1] = AB[1], AB[0]
    ans += -AB[1]
    AB[0] += AB[1]
    AB[1] = 0
    # 愚直に試してみて合うか合わないかでOK
    if sum(AB) == 0:
        print(ans)
    else:
        print(-1)
    exit()

# 実は加点が必要な人が2人以上いたら何やっても無理
cnt = 0
idx = -1
for i in range(N):
    if AB[i] < 0:
        cnt += 1
        idx = i
if cnt >= 2:
    print(-1)
    exit()

# 加点が必要な人に貪欲に足していって、途中でそれが2人以上になったらやっぱり無理
while cnt:
    cnt = 0
    nxt = -1
    # やりながら操作回数も数えておく
    ans += -AB[idx]
    for i in range(N):
        if i != idx:
            AB[i] += AB[idx]
            if AB[i] < 0:
                nxt = i
                cnt += 1
    if cnt >= 2:
        print(-1)
        exit()
    AB[idx] = 0
    idx = nxt

# ここまでで全員減点さえすればいい状態になる
# 1回の操作でみんな偶奇は反転するので、偶奇の違う人が1人でもいると絶対全0にはならない
se = set()
for i in range(N):
    se.add(AB[i]%2)
if len(se) != 1:
    print(-1)
    exit()

# 1回の操作で減少する値は総和に対しては一定(N-2ずつ)なので、それでぴったりいかないならダメ
if sum(AB) % (N-2) == 0:
    # 残りの操作回数は一意
    cnt = sum(AB) // (N-2)
    # 操作回数の偶奇とさっき取った現在の偶奇が合わないとダメ
    # and 操作回数よりも多く減らさないといけない人がいたら何やっても無理
    if cnt % 2 == list(se)[0] and max(AB) <= cnt:
        ans += cnt
        print(ans)
    else:
        print(-1)
else:
    print(-1)
