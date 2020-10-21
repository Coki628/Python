"""
・やっぱり通さないで放置は気持ち悪いので、頑張って通しておいた。テストケースとにらめっこして。
・条件判定が詰めきれてなかったのはダメだったとして、
　途中で変な空白が混ざったりするのは気付けないなぁ。。
"""

import sys
from string import digits, ascii_letters

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
sys.setrecursionlimit(10**9)
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

def check1(s, keyword):
        n = len(s)
        m = len(keyword)
        if n < m:
            return False
        # 語尾に記号が0〜3文字入る場合をチェック
        for i in range(4):
            if n-m-i < 0:
                break
            ss = s[n-m-i:]
            # キーワード部分
            sl = ss[:m]
            # 記号部分
            sr = ss[m:]
            if sl.lower() == keyword.lower():
                for sss in sr:
                    if sss in digits or sss in ascii_letters:
                        break
                else:
                    return True
        return False

def check2(S):
    for s in S:
        for ss in s:
            if ss in digits or ss in ascii_letters:
                return True
    return False

for line in sys.stdin:
    # splitするとよしなに空白が消されてしまうので、最初の空白だけ取得
    idx = line.find(' ')
    # 空白文字なし
    if idx == -1:
        print('WRONG!')
        continue
    name = line[:idx]
    S = line[idx+1:].strip()

    if name == 'digi':
        if check1(S, 'nyo'):
            print('CORRECT (maybe)')
        else:
            print('WRONG!')
    elif name == 'petit':
        if check1(S, 'nyu'):
            print('CORRECT (maybe)')
        else:
            print('WRONG!')
    elif name == 'gema':
        if check1(S, 'gema'):
            print('CORRECT (maybe)')
        else:
            print('WRONG!')
    elif name == 'piyo':
        if check1(S, 'pyo'):
            print('CORRECT (maybe)')
        else:
            print('WRONG!')
    elif name == 'rabi':
        if check2(S):
            print('CORRECT (maybe)')
        else:
            print('WRONG!')
    else:
        print('WRONG!')
