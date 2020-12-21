# -*- coding: utf-8 -*-

N = int(input())
S = input()

name = 'b'
names = [name]
# Nまでの文字数で名前リストを作成
while len(name) <= N:
    if len(names) % 3 == 1:
        name = 'a' + name + 'c'
    if len(names) % 3 == 2:
        name = 'c' + name + 'a'
    if len(names) % 3 == 0:
        name = 'b' + name + 'b'
    names.append(name)

try:
    print(names.index(S))
except ValueError:
    print(-1)
