"""
・自力ならず。。
・まず非本質で時間めっちゃ溶かした。何行来るか分からない入力がこんなに取るの大変だとは。。。
・結果的にはpythonだと、for line in sys.stdin: ってやればよしなに取れるみたい。
　手元でやると処理終了しないで止まっちゃうんだけど、コード投げるとちゃんと動く。。
・で、実際の判定処理はなんか色々グチャグチャやるんだけど、WAだしRE。。。
　もう疲れたからこれやんなくていいかな。。
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

def check1(S, keyword):
        s = S[-1]
        n = len(s)
        if n <= 2:
            return False
        for i in range(3):
            if n-3-i < 0:
                break
            ss = s[n-3-i:]
            sl = ss[:3]
            sr = ss[3:]
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
    name, *S = line.split()

    if not S:
        print('WRONG!')
        continue

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
