"""
参考：https://twitter.com/morio_prog/status/1276192777236144128
・自力ならず。これはまあしゃあないかな。
・桁和系、必要なところを全探索、地獄実装
・人の実装見てもピンと来ず、ツイートのヒントを手がかりに頑張って考えて実装した。
・だいたい要点はソース内コメントに書いたけど、全体的な話を。
・繰り上がりなしの方は末尾桁以外の桁和はKが決まれば一意。150以内を全部見ていい。
　構築方法としては最小値にしたいので、なるべく少ない桁数でx9999みたいにするのが最適。
　後は末尾桁の開始位置10箇所を全部試せばいい。これはまだいくらか楽。
・繰り上がりありがつらい。基本方針はなしのと同じ。以下、違う点を。
・まず何桁繰り上がりに使うかで全部見る。繰り上がりする桁の桁和が150以内、でやった。
　繰り上がりする桁の9999は最低でも1回は使うから、これだけで150オーバーなら確実に範囲外だろう、という推定。。
・桁の扱いが決まったら、x99899みたいにさっきのやつの間に8を挟んでそこまでは繰り上がり、の形にする。
・後はKの値と末尾桁の開始位置を決めれば、要素としていくつ繰り上がっていくつ上がらないか定まるので、
　それに従って繰り上がり分と繰り上がらない分と末尾桁分を足して答えに使う。
・大まかには組めても細部は詰めきれてなくて、落ちたテストケース検証しながらやっとACって感じ。。
・方針思いついてたとしても自力は無理だったぽいけど、強くなるためにはこういう重実装も通せたらいいな。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

def digit_sum(S):
    """ 桁和：O(|S|) """

    res = 0
    for s in S:
        res += int(s)
    return res

MAX = 150
ans = list2d(MAX+1, 10, -1)

# last[i][j] := 使用する末尾桁が[i,j]の時の末尾桁の桁和
last = list2d(10, 10, 0)
for i in range(10):
    for j in range(i, i+10):
        cnt = 0
        for k in range(i, j+1):
            k %= 10
            cnt += k
        last[i][j%10] = cnt

# 繰り上がりなし
# 末尾桁以外の和がi
for i in range(MAX+1):
    # 末尾桁以外の構築
    s = ''
    tmp = i
    while tmp-9 >= 0:
        s += '9'
        tmp -= 9
    if tmp:
        s += str(tmp)
    s = s[::-1]
    # Kの値
    for j in range(10):
        # 末尾桁以外はj+1回同じだけ増える
        cnt = i * (j+1)
        if cnt > MAX:
            break
        # 末尾桁の開始位置
        for k in range(10):
            if k+j >= 10:
                break
            res = cnt + last[k][k+j]
            if res <= MAX:
                if ans[res][j] == -1 or ans[res][j] > int(s+str(k)):
                    ans[res][j] = int(s+str(k))

# 繰り上がりあり
# 末尾桁以外の繰り上がり前の和がi
for i in range(MAX+1):
    x = 0
    while x*9 <= i:
        # 末尾桁以外の構築
        tmp = i
        s1 = '9' * x
        tmp -= x * 9
        # 繰り上がらない桁を挟む
        if tmp-8 >= 0:
            s1 += '8'
            tmp -= 8
        # 残りはまた9でOK
        while tmp-9 >= 0:
            s1 += '9'
            tmp -= 9
        if tmp:
            s1 += str(tmp)
        # ここが0(元々1桁で繰り上がりで2桁になるやつ)の場合
        if not s1:
            s1 = '0'
        s1 = s1[::-1]
        # 繰り上がり後
        s2 = str(int(s1)+1)
        # それぞれの桁和
        dsm1 = digit_sum(s1)
        dsm2 = digit_sum(s2)
        # Kの値
        for j in range(1, 10):
            # 末尾桁の開始位置
            for k in range(10):
                if k+j < 10:
                    continue
                # 繰り上がり前の個数
                a = 10 - k
                # 繰り上がり後の個数
                b = (j+1) - a
                cnt = dsm1 * a + dsm2 * b
                if cnt > MAX:
                    continue
                if k+j < 10:
                    continue
                res = cnt + last[k][(k+j)%10]
                if res <= MAX:
                    if ans[res][j] == -1 or ans[res][j] > int(s1+str(k)):
                        ans[res][j] = int(s1+str(k))
        x += 1

for _ in range(INT()):
    N, K = MAP()

    print(ans[N][K])
