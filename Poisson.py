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

gen = Psi_generator(k=3)
a = psi_from_y(0.25, 0.25, gen=gen)
print('a')
print(a)
b = y(0.25, 0.25, gen=gen)
print('b')
print(b)
c = g_wo_phi(0.25, 0.25, gen=gen)
print('c')
print(c)
d = g_wo_phi_prime(0.25, 0.25, gen=gen)
print('d')
print(d)
f = f_func(0.25, 0.25, gen=gen)
print('f')
print(f)
