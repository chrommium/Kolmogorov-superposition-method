import math
import pandas as pd
import numpy as np
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

# def I_from_func_adap(f, x_min, x_max, max_step):
# # def I_from_func(f, x_min, x_max, max_step):
#     step = max_step
#     p = 1 # степень 2: во сколько раз понижен шаг
#     if x_max - x_min < step:
#         return (x_max - x_min) * (f(x_min) + f(x_max)) / 2
#     res = 0
#     x = x_min
#     y1 = 0
#     y2 = 0
#     new_step = step
#     while x + step <= x_max + 10**-13:
        
#         step_not_found = True
#         step = new_step
#         y1 = f(x)

#         while step_not_found:
#             y2 = f(x + step)
#             y_m = max(abs(y1), abs(y2)) + 10**-13
#             # step = min(max_step/y_m, max_step)
#             p = math.ceil(math.log10(y_m*step))
#             p = max(0, p)
#             new_step = max_step * 10**-p
#             step = min(new_step, step)
#             if step <= new_step + 10**-13:
#                 step_not_found = False
#                 y2 = f(x + step)
        
#         x +=step 
#         res += (y1 + y2) * step / 2
#     y1 = y2
#     y2 = f(x_max)
#     res += (y1+y2)*(x_max-x)/2
#     return res

def I_Simpson_10(f, x_min, x_max):
    res = f(x_min) + f(x_min)
    step = (x_max-x_min) / 10
    for i in range(2, 10, 2):
        res += 2*f(x_min + step*i)
    for i in range(1, 10, 2):
        res += 4*f(x_min + step*i)
    return res * (x_max - x_min) / 30

def I_Simpson_iter(f, x_min, x_max, eps_abs, eps_rel):
    I_1 = I_Simpson_10(f, x_min, x_max)
    I_2 = I_Simpson_10(f, x_min, (x_min + x_max)/2) + I_Simpson_10(f, (x_min + x_max)/2, x_max)
    if abs(I_1 - I_2) < eps_abs or abs(I_2-I_1) / (max(abs(I_1), abs(I_2))+10**-13) < eps_rel:
        return I_2    
    return I_Simpson_iter(f, x_min, (x_min + x_max)/2, eps_abs, eps_rel) + \
            I_Simpson_iter(f, (x_min + x_max)/2, x_max, eps_abs, eps_rel)

def I_from_func_adap(f, x_min, x_max, step, eps_abs=10**-3, eps_rel=10**-3):
    big_step = step * 10
    if big_step > (x_max - x_min):
        return I_Simpson_iter(f, x_min, x_max, eps_abs, eps_rel)
    N = math.ceil((x_max - x_min)/step)
    big_step = (x_max - x_min)/N
    res = 0
    for i in range(N):
        res += I_Simpson_iter(f, x_min+big_step*i, x_min+big_step*(i+1), eps_abs, eps_rel)
    return res


# def f_test(x):
#     x0 = math.pi
#     sigma = 10**-6
#     return 1 / (((2*math.pi)**0.5)*sigma) * math.exp(-((x-x0)**2)/(2*sigma**2))

# def f_test_2(x):
#     c = 3*10**-9
#     x0 = 0.509089688889999
#     return 1/(math.pi * (1+((x-x0)/c)**2) * c)

# ABC_1 = I_from_func_adap(f_test, 0, 10, 10**-4, 10**-4, 10**-4)
# # ABC_1 = I_Simpson_iter(lambda x: x**3, 0, 10, 10**-4, 10**-4)
# # ABC_1 = I_Simpson_iter(f_test, 0, 10, 10**-4, 10**-4)
# # print('I_from_func_adapt = ' + str(ABC))
# print('I_from_func = ' + str(ABC_1))


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
    return I_from_func_adap(f, x_low(z), x_high(z), 10**-2) #gen.g**(-gen.k-1)

def I_g_wo_phi_prime(z):
    f = lambda x: g_wo_phi_prime(x,z,gen)
    return I_from_func_adap(f, x_low(z), x_high(z), 10**-2) #gen.g**(-gen.k-1)

def I_f_func(z):
    f = lambda x: f_func(x,z,gen)
    return I_from_func_adap(f, x_low(z), x_high(z), 10**-2) #gen.g**(-gen.k-1)


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
        self.tau1 = 0
        # self.h = [0, phi_prime_0*step]
        self.h = []
        self.I_h = []
        self.I_f_arr = []
        self.I_g_arr = []
        self.I_g_prime_arr = []
        # z = step # значение аргумента
        # n = 1 # номер ячейки в массиве, соответсвующей аргументы z
        
    def calc_integrals(self):
        tau = self.step
        for i in range(1, self.N-1):
            z = i*tau
            I_f = I_f_func(z + tau/2)
            I_g = I_g_wo_phi(z +tau/2)
            I_g_prime = (I_g_wo_phi(z + tau) - I_g_wo_phi(z)) / tau
            self.I_f_arr.append(I_f)
            self.I_g_arr.append(I_g)
            self.I_g_prime_arr.append(I_g_prime)
            print(i)

        self.I_f_arr.append(self.I_f_arr[-1])
        self.I_g_arr.append(self.I_g_arr[-1])
        self.I_g_prime_arr.append(self.I_g_prime_arr[-1])    
    
    def solve(self, h0):
        if len(self.I_g_arr) == 0:
            raise Exception('Empty array')
        
        self.h = [h0]
        self.I_h = [0]
        tau = self.step
        M = self.N * 10
        tau = tau / 10
        self.tau1 = tau
        l = len(self.I_f_arr)
        for i in range(1, M):
            j = min(i//10, l-1)
            k = i%10
            m = min(l-1, j+1)
            I_f_1 = self.I_f_arr[j]
            I_f_2 = self.I_f_arr[m]
            I_g_1 = self.I_g_arr[j]
            I_g_2 = self.I_g_arr[m]
            I_g_prime_1 = self.I_g_prime_arr[j]
            I_g_prime_2 = self.I_g_prime_arr[m]
            
            I_f = I_f_1 + k * (I_f_2 - I_f_1) / 10
            I_g = I_g_1 + k * (I_g_2 - I_g_1) / 10
            I_g_prime = I_g_prime_1 + k * (I_g_prime_2 - I_g_prime_1) / 10
            
            numerator = I_f - self.h[i-1] * I_g_prime
            self.h.append(self.h[i-1] + tau * numerator / I_g)
            self.I_h.append(self.I_h[i-1] + self.h[i-1] * tau)


    def phi(self, z):
        n = math.floor(z / self.tau1)
        if n < 0:
            return 0
        if n >= len(self.I_h):
            return self.I_h[-1]
        return self.I_h[n]


    def save_integrals(self, pref):
        pd.DataFrame(self.I_f_arr).to_csv('I_f'+pref+'.csv')
        pd.DataFrame(self.I_g_arr).to_csv('I_g'+pref+'.csv')
        pd.DataFrame(self.I_g_prime_arr).to_csv('I_g_prime'+pref+'.csv')
        pd.DataFrame(self.h).to_csv('h'+pref+'.csv')
        
    def load_integrals(self, pref):
        self.I_f_arr = pd.read_csv('I_f'+pref+'.csv').iloc[:,1]
        self.I_g_arr = pd.read_csv('I_g'+pref+'.csv').iloc[:,1]
        self.I_g_prime_arr = pd.read_csv('I_g_prime'+pref+'.csv').iloc[:,1]
        # return I_f_arr, I_g_arr, I_g_prime_arr
        

eul_1 = Euler_1(0.05, 10**-2)
# eul_1.calc_integrals()
# eul_1.save_integrals(pref = "2")
eul_1.load_integrals(pref = "2")

plt.plot(eul_1.I_f_arr, label='I_f')
plt.legend()
plt.show()
plt.plot(eul_1.I_g_arr, label='I_g')
plt.legend()
plt.show()
plt.plot(eul_1.I_g_prime_arr, label='I_g_prime')
plt.legend()
plt.show()


c1 = -10
c2 = 0
c = (c1 + c2) / 2
I_h_1 = -1
I_h_2 = 1
while (c2-c1 >= 10**-3):
    c = (c1 + c2) / 2
    eul_1.solve(c)
    I_h_mid = eul_1.I_h[-1]
    if I_h_mid*I_h_1 < 0:
        c2 = c
        I_h_2 = I_h_mid
    else:
        c1 = c
        I_h_1 = I_h_mid
print(c)

eul_1.solve(c)
plt.plot(eul_1.h)
plt.show()
plt.plot(eul_1.I_h)
plt.show()

def solution(x, y):
    return eul_1.phi(alpha1*gen.interpolated_psi_k(x) + alpha2*gen.interpolated_psi_k(y))


from matplotlib import cm

# init indexes
xs = [x for x in np.arange(0, 1.02, 0.02)]
ys = [x for x in np.arange(0, 1.02, 0.02)]
# functions results
zs1 = [solution(x, y) for x, y in zip(xs, ys)]

# plot_3d
zs_anlt = [[solution(x, y) for y in ys] for x in xs]
print(zs_anlt)
xs = np.array(xs)
ys = np.array(ys)
zs_anlt = np.array(zs_anlt)

# plot_3d analitycal
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
X, Y = np.meshgrid(xs, ys)
ax.plot_surface(X, Y, zs_anlt, antialiased=True, label='Kolmogorov solution')
# ax.text(0, 1, 0, 'Analytical solution', color='red')
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


# # z_arr = [i*10**-9 for i in range(5743000,5745000)]
# # z_arr = [i*10**-3 for i in range(0,1000)]
# # z_arr = [i*10**-7 for i in range(840500,841500)]
# # z_arr = [i*10**-7 for i in range(858000,859000)]
# g_arr = [g_wo_phi(0.484, i, gen) for i in z_arr]
# g_prime_arr = [g_wo_phi_prime(i, 0.96, gen) for i in z_arr]
# # phi_arr = [phi_from_y(i, 0.96, gen) for i in z_arr]
# r_arr = [r(0.484, i, gen) for i in z_arr]
# n_arr = [n_func(0.484, i, gen) for i in z_arr]
# y_arr = [y(0.484, i, gen) for i in z_arr]
# psi_from_y_arr = [psi_from_y(0.484, i, gen) for i in z_arr]

# plt.plot(z_arr, g_arr)
# plt.show()
# plt.plot(z_arr, g_prime_arr)
# plt.show()
# plt.plot(z_arr, r_arr)
# plt.show()
# plt.plot(z_arr, n_arr)
# plt.show()
# plt.plot(z_arr, y_arr)
# plt.show()

# print(psi_from_y(0.484, 0.00574400, gen)+1)
# print(psi_from_y(0.484, 0.00574401, gen)+1)
# print(psi_from_y(0.484, 0.00574402, gen)+1)
# print(psi_from_y(0.484, 0.00574403, gen)+1)
# print('----')
# # print(gen.backward_psi_k(0.5999999806212986))
# # print(gen.backward_psi_k(0.599999990))
# # print(gen.backward_psi_k(0.599999999))
# print(gen.backward_psi_k(0.5999999999999999))
# print(gen.backward_psi_k(0.59999999999999999))

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
