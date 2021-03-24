from Kolmogorov_Sprecher_Method import Psi_generator
import math
import matplotlib.pyplot as plt
from RK4 import Solver_RK4

alpha1 = 0.5
alpha2 = 0.5

# tested
def psi_from_y(x, z, gen):
    return (z/alpha2) - (alpha1/alpha2)*gen.interpolated_psi_k(x)

# tested
def y(x, z, gen):
    return gen.backward_psi_k(psi_from_y(x, z, gen))

# tested
def r(x, z, gen):
    return alpha2*gen.interpolated_psi_k_prime(y(x,z, gen))

# tested
def r_prime(x, z, gen):
    return alpha2*gen.interpolated_psi_k_prime2(y(x,z, gen)) / r(x,z,gen)

# tested
def n_func(x, z, gen):
    return (alpha1*gen.interpolated_psi_k_prime(x))**2 + (alpha2*gen.interpolated_psi_k_prime(y(x,z,gen)))**2

# tested
def n_prime(x, z, gen):
    factor1 = 2*(alpha2**2)*gen.interpolated_psi_k_prime(y(x,z,gen))
    factor2 = gen.interpolated_psi_k_prime2(y(x,z,gen)) / r(x,z,gen)
    return factor1 * factor2 

def g_wo_phi(x, z, gen):
    return n_func(x,z,gen) / r(x,z,gen)

# tested
def g_wo_phi_prime(x,z,gen):
    first_term = n_prime(x,z,gen)
    second_term = n_func(x,z,gen) * r_prime(x,z,gen) / r(x,z,gen)
    return (first_term - second_term) / r(x,z,gen)

def f_func(x,z, gen):
    return math.sin(math.pi*x) * math.sin(math.pi*y(x,z,gen))

# tested
def I_from_func(f, x_min, x_max, step):
    if x_max - x_min < step:
        return (x_max - x_min) * (f(x_min) + f(x_max)) / 2
    N = math.ceil((x_max - x_min) / step)
    step = (x_max - x_min) / N
    res = f(x_min) * step / 2
    for i in range(1, N):
        res += f(x_min+i*step) * step
    res += f(x_max) * step / 2
    return res

# arr = [i for i in range(100)]
# res = lambda x: I_from_func((lambda t: t**2), 0, x, 0.001)
# res_arr = [res(i*0.05) for i in arr]
# correct_res = lambda x: x**3/3
# correct_res_arr = [correct_res(i*0.05) for i in arr]
# plt.plot(arr, res_arr)
# plt.plot(arr, correct_res_arr)
# plt.legend()
# plt.show()

def x_low(z):
    if z < alpha2:
        return 0
    return gen.backward_psi_k((z-alpha2)/alpha1)

def x_high(z):
    if z < alpha1:
        return gen.backward_psi_k(z/alpha1)
    return 1

def I_g_wo_phi(z):
    f = lambda x: g_wo_phi(x,z,gen)
    return I_from_func(f, x_low(z), x_high(z), gen.g**(-gen.k))

def I_g_wo_phi_prime(z):
    f = lambda x: g_wo_phi_prime(x,z,gen)
    return I_from_func(f, x_low(z), x_high(z), gen.g**(-gen.k))

def I_f_func(z):
    f = lambda x: f_func(x,z,gen)
    return I_from_func(f, x_low(z), x_high(z), gen.g**(-gen.k))


# def ffunc(x, y):
#     return y**2/x
# sol = Solver_RK4(f=ffunc, x0=2, x_left=1, x_right=3, y_left=-1, y_right=-0.5, step=0.05)


# def generate_phi_from_y(x, gen):
#     return lambda z: psi_from_y(x,z,gen)
#     # def f_temp(z):
#     #     return psi_from_y(x, z, gen)


gen = Psi_generator(k=3, backward_eps=10**-8)
phi_from_y = lambda z: psi_from_y(0.25, z, gen)

def diff(f, step):
    def f_prime(x):
        return (f(x+step/2) - f(x-step/2))/step
    return f_prime

# z_array = [i*10**-7 for i in range(274000,276000)]
z_array = [i*5*10**-7 for i in range(2*27000,2*28000)]
y_array = [y(0.25, i, gen) for i in z_array]
y_prime = diff(lambda z: y(0.25, z, gen), 10**-7)
y_prime_array = [y_prime(i) for i in z_array]
r_array = [1/r(0.25, i, gen) for i in z_array]

# n_prime_array = [n_prime(0.25, i, gen) for i in z_array]
# n_prime_num = diff(lambda z: n_func(0.25, z, gen), 10**-7)
# n_prime_array_num = [n_prime_num(i) for i in z_array]
# plt.plot(z_array, n_prime_array)
# plt.plot(z_array, n_prime_array_num)
# plt.legend()
# plt.show()


g_wo_phi_array = [g_wo_phi(0.25, i, gen) for i in z_array]
g_wo_phi_prime_array = [g_wo_phi_prime(0.25, i, gen) for i in z_array]

g_wo_phi_prime_num = diff(lambda z: g_wo_phi(0.25, z, gen),10**-7)
g_wo_phi_prime_num_array = [g_wo_phi_prime_num(i) for i in z_array]


# plt.plot(z_array, y_array)
# plt.legend()
# plt.show()
# plt.plot(z_array, r_array)
# plt.legend()
# plt.show()
# plt.plot(z_array, y_prime_array)
# plt.legend()
# plt.show()
# plt.plot(z_array, g_wo_phi_array)
# plt.legend()
# plt.show()

plt.plot(z_array, g_wo_phi_prime_array)
# plt.legend()
# plt.show()
plt.plot(z_array, g_wo_phi_prime_num_array)
plt.legend()
plt.show()


# x_array = [i*10**-3 for i in range(0,1000)]
# make_array_psi = [gen.interpolated_psi_k(i) for i in x_array]
# plt.plot(x_array, make_array_psi)
# plt.legend()
# plt.show()

# print(gen.interpolated_psi_k(0.25))
# print(gen.psi_k(0.25))

s = 0.00508 - 0.205
# print(gen.interpolated_psi_k(s))
# print(gen.psi_k(s))

print(gen.backward_psi_k(s))
print(gen.backward_psi_k(s+1))


# gen = Psi_generator(k=3)
# a = psi_from_y(0.25, 0.25, gen=gen)
# print('a')
# print(a)
# b = y(0.25, 0.25, gen=gen)
# print('b')
# print(b)
# c = g_wo_phi(0.25, 0.25, gen=gen)
# print('c')
# print(c)
# d = g_wo_phi_prime(0.25, 0.25, gen=gen)
# print('d')
# print(d)
# f = f_func(0.25, 0.25, gen=gen)
# print('f')
# print(f)
