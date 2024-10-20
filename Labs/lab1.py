def sum(a: float, b: float) -> float: #useful for pytorch (it comp. into torchscript)
    return a + b

print(sum(5, 5))

# https://www.google.com/url?q=https%3A%2F%2Fnumpy.org%2Fdoc%2Fstable%2Fuser%2Fabsolute_beginners.html

# numpy is much faster than simple py
# foloseste _ la sfarsitul unei functii pentru a arta ca functia respectiva este implementata "inplace", de obicei pentru array-uri

import numpy as np

bruh = np.array([[1, 2, 3], [3, 2, 1], [1, 1, 1]])
print(bruh)

print(bruh.ndim)
print(bruh.shape)

a = np.ones(100)
print(a)

b = np.arange(100, 1001, 50)
print(b)