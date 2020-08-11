"""
参考：https://twitter.com/kyopro_friends/status/1292469735775285249
・多くの人がこの部分点取ってたので自分も取り組んだが詰めきれず。
・掛け算の状況を作るために、a=0~10,b=0~10を2次元にマッピングする感じにする。
・予めaがある数未満なら0、bがある数未満なら1、みたいな状態を作っておけば、
　不等号しか使えなくてもaかつbかどうかの判定をすることができる。
　この条件判定の作り方が見えなかった。(自分ノートの図も参照)
・あとは実装上、1~10を用意する時に、a,bどちらかだけを見てると、
　a,bどっちかが0だと正しく作れなくなるので、a+bをやってから1を作る。
・ちょっと変わった問題で面白かった。部分点、通せたらもっとよかったけどね。。
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

N = 10**5 * 2


zero = 1000
one = 1001
ab = 1002
ans = []
# aが0、bが0でも対応できるようにa+bを作ってから1を作る
ans.append(('+', 0, 1, ab))
ans.append(('<', zero, ab, one))

# 1~10を用意する
offset = 100
for m in range(10):
    for _ in range(m+1):
        ans.append(('+', offset+m, one, offset+m))

# A < 1~10 で0001111..を作る
offset2 = 200
for i in range(10):
    ans.append(('<', 0, offset+i, offset2+i))

# 1~10 < B で1110000..を作る
offset3 = 300
for i in range(10):
    ans.append(('<', offset+i-1, 1, offset3+i))

# !A < B の条件で A & B を作って成り立つ数を数える
offset4 = 400
for a in range(10):
    for b in range(10):
        ans.append(('<', offset2+a, offset3+b, offset4+a*10+b))
        ans.append(('+', 2, offset4+a*10+b, 2))

print(len(ans))
for a in ans:
    print(*a)

# def check(a, b):
#     res = [0] * N
#     res[0] = a
#     res[1] = b
#     for op, i, j, k in ans:
#         if op == '+':
#             res[k] = res[i] + res[j]
#         else:
#             res[k] = res[i] < res[j]
#     print(a, b, res[2])
# check(0, 3)

# for i in range(11):
#     for j in range(11):
#         check(i, j)
