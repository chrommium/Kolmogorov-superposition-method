from Kolmogorov_Sprecher_Method import Psi_generator
import math
import matplotlib.pyplot as plt

alpha1 = 0.5
alpha2 = 0.5

def psi_from_y(x, z, gen):
    return (z/alpha2) - (alpha1/alpha2)*gen.interpolated_psi_k(x)

def y(x, z, gen):
    return gen.backward_psi_k(psi_from_y(x, z, gen))

def r(x, z, gen):
    return alpha2*gen.interpolated_psi_k_prime(y(x,z, gen))

def g_wo_phi(x,z,gen):
    numerator = (alpha1*gen.interpolated_psi_k_prime(x))**2 + (alpha2*gen.interpolated_psi_k_prime(y(x,z,gen)))**2
    return numerator / r(x,z,gen)

def g_wo_phi_prime(x,z,gen):
    first_term_numerator = 2 * alpha2**2 * gen.interpolated_psi_k_prime(y(x,z,gen))
    second_term_numerator = g_wo_phi(x,z,gen) / r(x,z,gen)
    return (first_term_numerator + alpha2 * second_term_numerator / r(x,z,gen)) * gen.interpolated_psi_k_prime2(y(x,z,gen))

def f_func(x,z, gen):
    return math.sin(math.pi*x) * math.sin(math.pi*y(x,z,gen))

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
z_array = [i*10**-6 for i in range(27000,28000)]
y_array = [y(0.25, i, gen) for i in z_array]
y_prime = diff(lambda z: y(0.25, z, gen), 10**-7)
r_array = [1/r(0.25, i, gen) for i in z_array]
y_prime_array = [y_prime(i) for i in z_array]
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
