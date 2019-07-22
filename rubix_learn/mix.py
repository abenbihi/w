
import numpy as np


mix_l = [l.split("\n")[0] for l in open('mix.txt').readlines() ]
mix_num = len(mix_l)



random_order = np.arange(mix_num)
np.random.shuffle(random_order)

while True:
    for idx in random_order:
        str_ret = input(mix_l[idx])
    np.random.shuffle(random_order)

