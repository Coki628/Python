"""
・実装おつ。
・初期位置に画面端がなかったのがせめてもの優しさ。でもだるい。
　とりあえず上まで言って、1行目を終わらせつつ左上端に行って、そこから1行ずつジグザグに下りてく。
　みんなもっと簡単な実装組んでそうだよなー。
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

W, H, sx, sy = MAP()

ans = [(sx, sy)]
ans.append((sx, 1))

curx = sx
while curx+1 <= W:
    curx += 1
    ans.append((curx, 1))
curx = sx - 1
ans.append((curx, 1))
while curx-1 >= 1:
    curx -= 1
    ans.append((curx, 1))
cury = 1
while cury+1 <= H:
    cury += 1
    ans.append((curx, cury))
    if cury % 2 == 0:
        while curx+1 <= W:
            curx += 1
            if (curx, cury) == (sx, sy):
                continue
            ans.append((curx, cury))
    else:
        while curx-1 >= 1:
            curx -= 1
            if (curx, cury) == (sx, sy):
                continue
            ans.append((curx, cury))

for x, y in ans:
    print(x, y)
