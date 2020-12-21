# -*- coding: utf-8 -*-

"""
自力AC！
"""

A, B = map(int, input().split())

# 上半分を白い面、下半分を黒い面
grid = [['.'] * 100 for i in range(100)]
for i in range(50, 100):
	for j in range(100):
		grid[i][j] = '#'
A -= 1
B -= 1
# 白い面に黒点
for i in range(1, B//50*2+1, 2):
	for j in range(100):
		if j % 2 == 0:
			grid[i][j] = '#'
for i in range(0, B%50*2, 2):
	grid[B//50*2+1][i] = '#'
# 黒い面に白点
for i in range(51, A//50*2+51, 2):
	for j in range(100):
		if j % 2 == 0:
			grid[i][j] = '.'
for i in range(0, A%50*2, 2):
	grid[A//50*2+51][i] = '.'

print(100, 100)
for i in range(100):
	print(''.join(grid[i]))
