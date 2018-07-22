# -*- coding: utf-8 -*-

S,c = map(int, input().split())

# c→Sは変換できるので、合算して考える
if c >= S*2:
    print((S*2 + c) // 4)
# S→cには変換できないので、完全にCの数次第
else:
    print(c // 2)