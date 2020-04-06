import numpy as np
from collections import defaultdict
perst_dict = {i: 0 for i in range(10)}
n = 1000000
for i in range(10, n):
    perst_dict[i] = 1 + perst_dict[np.prod([int(j) for j in str(i)])]

inverse_dict = defaultdict(list)
for k, v in perst_dict.items():
    inverse_dict[v].append(k)
res_dict = {k: min(v) for k, v in inverse_dict.items()}
print(res_dict)