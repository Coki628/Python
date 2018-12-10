# -*- coding: utf-8 -*-

S = input()

ans = float('inf')
for i in range(len(S)-2):
    ans = min(abs(int(S[i:i+3]) - 753), ans)
print(ans)
