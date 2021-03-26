import math
from Kolmogorov_Sprecher_Method import Psi_generator
import matplotlib.pyplot as plt

gen = Psi_generator(k=2, backward_eps=10**-8)
phi_from_y = lambda z: psi_from_y(0.25, z, gen)
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
# def I_from_func(f, x_min, x_max, step):
#     if x_max - x_min < step:
#         return (x_max - x_min) * (f(x_min) + f(x_max)) / 2
#     N = math.ceil((x_max - x_min) / step)
#     step = (x_max - x_min) / N
#     res = f(x_min) * step / 2
#     for i in range(1, N):
#         res += f(x_min+i*step) * step
#     res += f(x_max) * step / 2
#     return res

 # def I_from_func_adap(f, x_min, x_max, min_step):
def I_from_func(f, x_min, x_max, min_step):
    step = min_step
    p = 1 # степень 2: во сколько раз понижен шаг
    if x_max - x_min < step:
        return (x_max - x_min) * (f(x_min) + f(x_max)) / 2
    res = 0
    x = x_min
    y1 = 0
    y2 = 0
    new_step = step
    while x + step <= x_max + 10**-13:
        
        step_not_found = True
        step = new_step
        while step_not_found:
            y1 = f(x)
            y2 = f(x + step)
            y_m = max(abs(y1), abs(y2))
            # step = min(min_step/y_m, min_step)
            p = math.ceil(math.log2(y_m))
            new_step = min_step * 2**-p
            step = min(new_step, step)
            if step <= new_step + 10**-13:
                step_not_found = False

        res += (y1 + y2) * step / 2
    y1 = y2
    y2 = f(x_max)
    res += (y1+y2)*(x_max-x)/2
    return res

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
    return I_from_func(f, x_low(z), x_high(z), gen.g**(-gen.k-1))

def I_g_wo_phi_prime(z):
    f = lambda x: g_wo_phi_prime(x,z,gen)
    return I_from_func(f, x_low(z), x_high(z), gen.g**(-gen.k-1))

def I_f_func(z):
    f = lambda x: f_func(x,z,gen)
    return I_from_func(f, x_low(z), x_high(z), gen.g**(-gen.k))


class Euler:
    '''метод Эйлера для phi(z)'''
    def __init__(self, phi_prime_0, step):
        self.step = step
        self.N = math.ceil((alpha1 + alpha2) / step)
        self.step = (alpha1 + alpha2) /self.N
        self.phi = [0, phi_prime_0*step]
        self.I_f_arr = []
        self.I_g_arr = []
        self.I_g_prime_arr = []
        # z = step # значение аргумента
        # n = 1 # номер ячейки в массиве, соответсвующей аргументы z
        tau = self.step
        for i in range(1, self.N-1):
            z = i*self.step
            I_f = I_f_func(z)
            I_g = I_g_wo_phi(z)
            I_g_prime = I_g_wo_phi_prime(z)
            A_1 = (I_g_prime / (2 * tau)) - (I_g / tau**2)
            B_1 = 2 * I_g / tau**2
            C_1 = (I_g / tau**2) + (I_g_prime / (2 * tau))
            numerator = I_f + self.phi[i]*B_1 + self.phi[i-1]*A_1
            self.phi.append(numerator/C_1)
            self.I_f_arr.append(I_f)
            self.I_g_arr.append(I_g)
            self.I_g_prime_arr.append(I_g_prime)
            print(i)

# eul = Euler(0.05, 10**-2)
# eul
# plt.plot(eul.phi[:50])
# plt.show()
# plt.plot(eul.I_f_arr, label='I_f')
# plt.legend()
# plt.show()
# plt.plot(eul.I_g_arr, label='I_g')
# plt.legend()
# plt.show()
# # plt.plot(eul.I_g_prime_arr[:50], label='I_g_prime')
# # plt.legend()
# # plt.show()

class Euler_1:
    '''метод Эйлера для phi(z)'''
    def __init__(self, phi_prime_0, step):
        self.step = step
        self.N = math.ceil((alpha1 + alpha2) / step)
        self.step = (alpha1 + alpha2) /self.N
        self.h = [0, phi_prime_0*step]
        self.I_f_arr = []
        self.I_g_arr = []
        self.I_g_prime_arr = []
        # z = step # значение аргумента
        # n = 1 # номер ячейки в массиве, соответсвующей аргументы z
        tau = self.step
        for i in range(1, self.N-1):
            z = i*self.step
            I_f = I_f_func(z)
            I_g = I_g_wo_phi(z)
            I_g_prime = I_g_wo_phi_prime(z)

            numerator = I_f - self.h[i] * I_g_prime
            # A_1 = (I_g_prime / (2 * tau)) - (I_g / tau**2)
            # B_1 = 2 * I_g / tau**2
            # C_1 = (I_g / tau**2) + (I_g_prime / (2 * tau))
            # numerator = I_f + self.h[i]*B_1 + self.h[i-1]*A_1
            self.h.append(self.h[i-1] + 2 * tau * numerator / I_g)
            self.I_f_arr.append(I_f)
            self.I_g_arr.append(I_g)
            self.I_g_prime_arr.append(I_g_prime)
            print(i)


eul_1 = Euler_1(0.05, 2*10**-2)
eul_1
plt.plot(eul_1.h[:70])
plt.show()
plt.plot(eul_1.I_f_arr, label='I_f')
plt.legend()
plt.show()
plt.plot(eul_1.I_g_arr, label='I_g')
plt.legend()
plt.show()
plt.plot(eul_1.I_g_prime_arr[:50], label='I_g_prime')
plt.legend()
plt.show()

# x_arr = [i for i in range(0,1000)]
# x_low_arr = [x_low(i*10**-3) for i in x_arr]
# x_high_arr = [x_high(i*10**-3) for i in x_arr]
# plt.plot(x_low_arr, label='x_low')
# plt.legend()
# plt.plot(x_high_arr, label='x_high')
# plt.legend()
# plt.show()


# x_arr = [i for i in range(9500,10500)]
# x_arr = [i*10**-3 for i in range(0,1000)]
# g_wo_phi_array = [g_wo_phi(i, 0.1, gen) for i in x_arr]
# g_wo_phi_prime_array = [g_wo_phi_prime(i, 0.1, gen) for i in x_arr]

# plt.plot(x_arr, g_wo_phi_array, label='g_wo_phi')
# plt.legend()
# plt.show()
# plt.plot(x_arr, g_wo_phi_prime_array, label='g_wo_phi_prime')
# plt.legend()
# plt.show()


# z_arr = [i*10**-9 for i in range(5743000,5745000)]
# z_arr = [i*10**-7 for i in range(840500,841500)]
z_arr = [i*10**-7 for i in range(858000,859000)]
g_arr = [g_wo_phi(0.484, i, gen) for i in z_arr]
g_prime_arr = [g_wo_phi_prime(i, 0.96, gen) for i in z_arr]
# phi_arr = [phi_from_y(i, 0.96, gen) for i in z_arr]
r_arr = [r(0.484, i, gen) for i in z_arr]
n_arr = [n_func(0.484, i, gen) for i in z_arr]
y_arr = [y(0.484, i, gen) for i in z_arr]
psi_from_y_arr = [psi_from_y(0.484, i, gen) for i in z_arr]

plt.plot(z_arr, g_arr)
plt.show()
plt.plot(z_arr, g_prime_arr)
plt.show()
plt.plot(z_arr, r_arr)
plt.show()
plt.plot(z_arr, n_arr)
plt.show()
plt.plot(z_arr, y_arr)
plt.show()

# print(psi_from_y(0.484, 0.00574400, gen)+1)
# print(psi_from_y(0.484, 0.00574401, gen)+1)
# print(psi_from_y(0.484, 0.00574402, gen)+1)
# print(psi_from_y(0.484, 0.00574403, gen)+1)
print('----')
# print(gen.backward_psi_k(0.5999999806212986))
# print(gen.backward_psi_k(0.599999990))
# print(gen.backward_psi_k(0.599999999))
print(gen.backward_psi_k(0.5999999999999999))
print(gen.backward_psi_k(0.59999999999999999))

# print(gen.backward_psi_k(0.6000000006212984))
# print(gen.backward_psi_k(0.6000000206212985))


# plt.plot(z_arr, psi_from_y_arr)
# plt.show()








# a = 0.009490435734784478
# # b = 0.5905095642652155
# A = 21.358927193507377
# # C = 0.5540509564265216
# B = 0.1
# eta = 0.0918980871469568
# x8_ = 0.581019128530431
# y8_ = 0.5081019128530431
# x10_ = 0.6
# y10_ = 0.6000000000000001

# b = a
# C = B*a + A*a*gen.stand_norm/2

# def F(ksi):
#     return gen.ihat(ksi, A, a, B, b, C)

# def dscrp(ksi):
#     return F(ksi) + B * ksi - eta 

# def dscrp_1(ksi):
#     return F(ksi) - eta 

# ksi_arr = [0.01 + i*10**-5 for i in range(0,1000)]
# F_arr = [F(i) + y8_ for i in ksi_arr]
# dscrp_arr = [dscrp(i) for i in ksi_arr]
# dscrp_1_arr = [dscrp_1(i) for i in ksi_arr]

# plt.plot(ksi_arr, F_arr)
# plt.show()
# plt.plot(ksi_arr, dscrp_arr, label = 'old')
# plt.plot(ksi_arr, dscrp_1_arr)
# plt.legend()
# plt.show()
