# -*- coding: utf-8 -*-

T = int(input())

for i in range(T):
    N, K = map(int, input().split())
    mmf = N // K
    ans = ''
    # mmfに基づいて必要な分
    for c in range(ord('a'), ord('a')+K):
        ans += chr(c) * mmf
    # 余りは適当にa
    ans += 'a' * (N % K)
    print(ans)
