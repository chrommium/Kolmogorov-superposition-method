from Kolmogorov_Sprecher_Method import Psi_generator
import math
import matplotlib.pyplot as plt

print('aasads')
gen2 = Psi_generator(k=3)
start = 0
end = 1000
g_ = 10 ** - 3
array_back = [gen2.backward_psi_k(i*(g_)) for i in range(start, end)]


x_ax = [i*(g_) for i in range(start, end)]
plt.plot(x_ax, array_back, label='backward_psi_k, k = 3')
plt.legend()
plt.show()
