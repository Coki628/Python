"""
・なんとか自力AC
・制約的に何かの貪欲なんだろうなとは思いつつさっぱりいい方針が出ないまま時間が過ぎる。
・状況を見ると通され過ぎなので、絶対難しくない何かだと確信する。
・なんとなく「でかい方から適当に詰めたらいいんじゃね？」となってやってみる。AC。
　ひどい。完全に未証明。お気持ちすらない。。
・一夜明けて、落ち着いてよくよく考えてみると、
　・Wより大きいのはそもそも入らないので無理
　・W以下半分以上のやつは1つ入ればその時点でOK
　・半分未満のやつは単体で半分以上一気に増やすことがないから、
　　順番に足していって足りるならW以下半分以上に収まる瞬間が絶対あるし、
　　足りないなら最後まで足りない。
　って感じか。。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N, W = MAP()
    target = ceil(W, 2)

    AI = [(a, i) for i, a in enumerate(LIST())]
    AI.sort(reverse=1)

    ans = []
    cur = 0
    for i in range(N):
        if cur+AI[i][0] <= W:
            cur += AI[i][0]
            ans.append(AI[i][1]+1)
            if cur >= target:
                break
    else:
        print(-1)
        continue
    print(len(ans))
    print(*ans)
