# -*- coding: utf-8 -*-

# こういう数学っぽい解き方できるようになれたらなー。

S = list(input())
N = int(input())

a = (N-1) // 5
b = (N-1) % 5
print(S[a] + S[b])