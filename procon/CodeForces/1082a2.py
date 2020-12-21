# -*- coding: utf-8 -*-

t = int(input())

ans = [0] * t
for i in range(t):
    n, x, y, d = map(int, input().split())
    visited = {}
    cnts = []
    # なるべく近づく
    cnt = 0
    cur = x
    while True:
        if visited.get(cur):
            cnts.append(float('inf'))
            break
        if cur < y:
            cur += d
        elif x > y:
            cur -= d
        elif cur == y:
            cnts.append(cnt)
            break
        cnt += 1
        visited[cur] = True
    # いったん1ページ目に来て進む
    cnt = 0
    cur = x
    while True:
        if cur == y:
            cnts.append(cnt)
            break
        if cur > 0:
            cur -= d
            cnt += 1
        else:
            if (y-1) % d == 0:
                cnts.append(cnt + (y-1) // d)
            else:
                cnts.append(float('inf'))
            break
    # いったん最終ページに来て戻る
    cnt = 0
    cur = x
    while True:
        if cur == y:
            cnts.append(cnt)
            break
        if cur <= n:
            cur += d
            cnt += 1
        else:
            if (n-y) % d == 0:
                cnts.append(cnt + (n-y) // d)
            else:
                cnts.append(float('inf'))
            break
    # 3パターンのうち最小の値
    if min(cnts) == float('inf'):
        print(-1)
    else:
        print(min(cnts))
