"""
・自力AC
・実装おつ。頑張って条件作る。
・全部2つずつのやつは別処理でいいとして、3つ揃えるのが同じ3つでも続き番号3つでもいいのが厄介。
・先に続き番号集めて残りで同じの3つにするのと、順番逆にやったやつ、
　それぞれを2つ揃えにする番号1つを決め打ってやった。
・サンプル合って、これでWAならどうしよっかなって思ってとりあえず投げたら無事AC。。
"""

import sys
from collections import Counter

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

S = list(map(int, input()))
N = len(S)

# 5個以上あるのはダメ
def check1(C):
    for v in C.values():
        if v >= 5:
            return False
    return True

# 先に続き番号取って、残りを同じ番号で揃える
def check2(C, x):
    for i in range(1, 8):
        while C[i] > 0 and C[i+1] > 0 and C[i+2] > 0:
            if x in [i, i+1, i+2] and C[x]-1 < 2:
                break
            C[i] -= 1
            C[i+1] -= 1
            C[i+2] -= 1
    tmp = []
    for k in C:
        if not C[k]:
            tmp.append(k)
        if k == x:
            if C[k] % 3 == 2:
                tmp.append(k)
        else:
            if C[k] % 3 == 0:
                tmp.append(k)
    for k in tmp:
        del C[k]
    if not C:
        return True
    else:
        return False

# 同じ番号を3つずつ取って、残りで続き番号を取る
def check3(C, x):
    for k in C:
        if k == x:
            while C[k]-3 >= 2:
                C[k] -= 3
        else:
            C[k] %= 3
    for i in range(1, 8):
        while C[i] > 0 and C[i+1] > 0 and C[i+2] > 0:
            if x in [i, i+1, i+2] and C[x]-1 < 2:
                break
            C[i] -= 1
            C[i+1] -= 1
            C[i+2] -= 1
    tmp = []
    for k in C:
        if not C[k]:
            tmp.append(k)
    for k in tmp:
        del C[k]
    if len(C) == 1:
        return True
    else:
        return False

# 全部2つのやつのチェック
def check4(C):
    for v in C.values():
        if v != 2:
            return False
    return True

ans = []
for num in range(1, 10):
    C = Counter(S+[num])

    if not check1(C.copy()):
        continue
    if check4(C.copy()):
        ans.append(num)
        continue
    # 唯一2つ組にする番号を決め打って全部やる
    for k in C:
        if check2(C.copy(), k) or check3(C.copy(), k):
            ans.append(num)
            break
for a in ans:
    print(a)
