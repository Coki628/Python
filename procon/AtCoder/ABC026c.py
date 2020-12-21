# -*- coding: utf-8 -*-

class Worker:
    def __init__(self, id, sv):
        self.id = id
        self.sv = sv
        self.salary = 0

N = int(input())
workers = []
# 社長
workers.append(Worker(1, 0))
# 他の社員
for i in range(N-1):
    workers.append(Worker(i+2, int(input())))

# 後ろから決めていけばうまくいく
workers.reverse()

for i in range(N):
    # 部下を探して自分の給料を確定
    max_sub = 0
    min_sub = float('inf')
    # 内側ループは自分より前だけやればいい
    for j in range(i):
        if workers[j].sv == workers[i].id:
            max_sub = max(workers[j].salary, max_sub)
            min_sub = min(workers[j].salary, min_sub)
    # 部下0人の調整
    if min_sub == float('inf'):
        min_sub = 0
    workers[i].salary = max_sub + min_sub + 1

# 配列の最後には社長の給料
print(workers[-1].salary)