from scipy import stats, optimize, interpolate
import numpy as np

p = [0.8, 0.2, 0.3]
p = np.array(p)
res = stats.bernoulli(p)

print()