
# python scripts/generate.py --cfg config.yaml --mode sample --model_pth ./ckp/model_linear.pth --res_path ./res


# Original array

import numpy as np
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

# New x values (more points)
x_new = np.linspace(x.min(), x.max(), 10)  # 10 points instead of 5

# Interpolating y values
y_new = np.interp(x_new, x, y)



print("Original x:", x)
print("Original y:", y)
print("New x:", x_new)
print("Interpolated y:", y_new)