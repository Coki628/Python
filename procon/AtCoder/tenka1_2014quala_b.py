# -*- coding: utf-8 -*-

"""
・自力AC
・実装おつ系、絶対あっとこのRatedじゃ出なそうなやつ。
・0.5があるけど、半開区間とみなせば実質小数は意識しなくてよし。
・後は時間経過による各種イベントの管理をしっかり追えばOK。
・なぜか制約のNが100万で、pythonTLE,pypyAC0.6秒。10万でもいい気がするが。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

S = input()

ret = [0] * 5
trgn = [-1] * 5
trgc = -1
wait = 0
com = ans = 0

# 時間経過による各種フラグのカウント
def timer():
    global trgc, wait
    for i in range(5):
        ret[i] -= 1
        trgn[i] -= 1
    trgc -= 1
    wait -= 1

# 通常投げができるか確認
def check1():
    for i in range(5):
        if ret[i] <= 0:
            return True
    return False

# 通常投げをする
def atk1():
    global ans
    for i in range(5):
        if ret[i] <= 0:
            ret[i] = 7
            trgn[i] = 2
            break
    ans += 10 * (1+com//10*0.1)

# ため投げができるか確認
def check2():
    cnt = 0
    for i in range(5):
        if ret[i] <= 0:
            cnt += 1
    return cnt >= 3

# ため投げをする
def atk2():
    global ans, trgc, wait
    cnt = 0
    for i in range(5):
        if ret[i] <= 0:
            ret[i] = 9
            cnt += 1
        if cnt == 3:
            trgc = 4
            wait = 3
            break
    ans += 50 * (1+com//10*0.1)

# コンボ成立確認
def check3():
    cnt = 0
    for i in range(5):
        if trgn[i] == 0:
            cnt += 1
    if trgc == 0:
        cnt += 1
    return cnt

for cmd in S:
    # コンボ成立確認
    com += check3()
    # ため中 or '-' は何もしない
    if wait > 0 or cmd == '-':
        timer()
        continue
    # 袋の残数を確認して、通常投げ
    if cmd == 'N' and check1():
        atk1()
    # 袋の残数を確認して、ため投げ
    elif cmd == 'C' and check2():
        atk2()
    # 時間経過による各種フラグのカウント
    timer()
print(int(ans))
