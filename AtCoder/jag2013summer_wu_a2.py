"""
・Shortest用
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def MAP(): return map(int, input().split())

N, M = MAP()
TS = []
for i in range(N):
    s, t = MAP()
    if t < s:
        t += M
    TS.append((t, s))
TS.sort()

nxt = list2d(17, N, -1)

l = r = 0
while l < N:
    while r < N and TS[l][0] > TS[r][1]:
        r += 1
    if r < N:
        nxt[0][l] = r
    l += 1

for k in range(1, 17):
    for i in range(N):
        nxt[k][i] = nxt[k-1][nxt[k-1][i]]   

ans = 0
for i in range(N):
    _, s = TS[i]
    cnt = cur = 0
    for k in range(16, -1, -1):
        if nxt[k][i] != -1:
            t, _ = TS[nxt[k][i]]
            if cur + (t - s) <= M:
                cnt += 2 ** k
                cur += t - s
                s = t
                i = nxt[k][i]
    ans = max(ans, cnt)
print(ans + 1)
