import math
import matplotlib.pyplot as plt
# import numpy as np


class Psi_generator:
    def __init__(self, n=2, g=10, k=4, backward_eps = 10**-7):
        self.n = n
        self.g = g
        self.k = k
        self.default_step = 0.001
        self.tau = self.g ** - (k + 2)
        self.eps_factor = 3
        self.precision = 4
        self.interp_list = [] # начала и концы линейных интервалов
        self.dict_param = {} # параметры функций-шапочек
        self.dict_funct = {} # запомниненные проинтегрированные функции-шапочки
        self.dict_prime = {} # запомниненные проивзодные
        self.dict_prime2 = {} # запомниненные вторые проивзодные
        self.dict_rev = {}
        self.B = self.g ** - (self.a_(self.k, 0) - self.k)
        self.stand_hat = self.generate_hat(1, 1, 0, 0)
        self.i_stand_hat, self.back_i_stand_hat, self.F_max = self.integrated_f(-1, 1, self.stand_hat, self.default_step)
        self.stand_norm = self.i_stand_hat(1)
        self.delta_x = self.g**-k
        self.y_step = self.g**-(k+1)
        self.backward_eps = backward_eps
        self.interval_index_for_y = [0]

        for i in range(self.g**(k-1)):
            x0 = i*self.g**-(k-1)
            x8 = x0 + (self.g-2)*(self.delta_x)

            y0 = self.psi_k(x0)
            y8 = self.psi_k(x8)

            self.interp_list.append((x0, y0))
            self.interp_list.append((x8, y8))
        self.interp_list.append((1, 1))

        j = 0
        for i in range(1, math.ceil(1/self.y_step) + 1):
            y = self.y_step * i
            while (j+1 < len(self.interp_list)) and (self.interp_list[j+1][1] <= y + 10**-13):
                j += 1
            self.interval_index_for_y.append(j)

    def lin_interp_psi_k(self, x):
        j = math.floor(x * self.g**(self.k-1))
        le = x * self.g**self.k - j*self.g
        # проеверяем, находимся ли на линейном участке или нет.
        # если на линейном, то интерполируем линейно

        if le < 8:
            x0, y0 = self.interp_list[j*2]
            return y0 + (x-x0) * self.B #func_linear(x, x0, y0, self.B)
        else:
            x8, y8 = self.interp_list[j*2+1]
            x10, y10 = self.interp_list[j*2+2]

            x9 = (x8 + x10) / 2
            y9 = self.psi_k(x9)

            if le == 9:
                return y9
            if le < 9:
                return y8 + (x-x8)*(y9-y8)/(x9-x8)

            return y9 + (x-x9)*(y10-y9)/(x10-x9)

    def interpolated_psi_k_all(self, x, func_linear, func_hat):

        j = math.floor(x * self.g**(self.k-1))
        le = x * self.g**self.k - j*self.g
        # проеверяем, находимся ли на линейном участке или нет.
        # если на линейном, то интерполируем линейно

        if le < 8:
            x0, y0 = self.interp_list[j*2]
            # y = y0 + (x-x0) * self.B
            # return y
            return func_linear(x, x0, y0, self.B)
        else:
            x8, y8 = self.interp_list[j*2+1]
            x10, y10 = self.interp_list[j*2+2]
            if (x8, x10) in self.dict_param:
                a, b, A, C = self.dict_param[(x8, x10)]
                # return self.ihat(x, A, a, self.B, b, C)
                return func_hat(x, A, a, self.B, b, C)
            else:
                x9 = (x8 + x10) / 2
                y9 = self.psi_k(x9)
                step = self.g**-(self.k) * self.default_step
                ksi = self.find_ksi(self.delta_x, x8, y8, y9, y10, step, step)
                ihd = self.gen_ihat_discrep(self.delta_x, x8, y8, y9, y10, step)
                _, a, b, A, C = ihd(ksi)
                self.dict_param[(x8, x10)] = a, b, A, C
                # return self.ihat(x, A, a, self.B, b, C)
                return func_hat(x, A, a, self.B, b, C)

    def linear(self, x, x0, y0, B):
        return y0 + (x - x0) * B

    def const(self, x, x0, y0, B):
        return B

    def zero(self, x, x0, y0, B):
        return 0

    def hat_prime(self, x, A, a, B, b, C):
        arg = (x - b) / a
        return - ((2 * arg / a) / (1 - arg ** 2) ** 2) * self.hat_wo_b(x, A, a, B, b)

    def hat_with_all_agrs(self, x, A, a, B, b, C):
        return self.hat_with_b(x, A, a, B, b)

    def interpolated_psi_k(self, x):
        whole_part = math.floor(x)
        x -= whole_part
        return whole_part + self.interpolated_psi_k_all(x, self.linear, self.ihat)

    def interpolated_psi_k_prime(self, x):
        whole_part = math.floor(x)
        x -= whole_part
        return self.interpolated_psi_k_all(x, self.const, self.hat_with_all_agrs)

    def interpolated_psi_k_prime2(self, x):
        whole_part = math.floor(x)
        x -= whole_part
        return self.interpolated_psi_k_all(x, self.zero, self.hat_prime)

    def lin_interpolated_psi_k_prime(self, x, delta_x):
        return (self.lin_interp_psi_k(x+delta_x/2) - self.lin_interp_psi_k(x-delta_x/2)) / delta_x

    def lin_interpolated_psi_k_prime2(self, x, delta_x):
        prime2 = self.lin_interpolated_psi_k_prime(x+delta_x/2, delta_x) - self.lin_interpolated_psi_k_prime(x-delta_x/2, delta_x)
        return prime2 / delta_x

    def a_(self, r, m_r):
        return (self.n**(r-m_r)-1) / (self.n-1)

    def i_corner(self, i_r, r):
        if r == 1:
            return 0
        if i_r == self.g-1:
            return 1
        return 0

    def i_sqbr(self, i_r, r):
        if r == 1:
            return 0
        if i_r >= self.g-2:
            return 1
        return 0

    def i_wave(self, i_r, r):
        return i_r-(self.g-2)*self.i_corner(i_r, r)

    def m_(self, r, i):
        sum_ = 1
        for s in range(1, r):
            product_ = 1
            for j in range(s, r):
                product_ *= self.i_sqbr(i[j], j)
            sum_ += product_
        return self.i_corner(i[r], r)*sum_

    def psi_k(self, x):

        x_ = math.floor(x)
        x = x-x_
        n_ = x*(self.g**self.k) + self.g**(-self.precision)
        n_ = math.floor(n_)

        i = []
        for j in range(0, self.k):
            i.append(n_ % self.g)
            n_ = n_//self.g
        # add fake value
        i.append(-1)
        i.reverse()

        s = 0
        for r in range(1, self.k+1):
            m_r = self.m_(r, i)
            s += self.i_wave(i[r], r)*(self.g**(-self.a_(r, m_r)))*(2**(-m_r))

        return s+x_


    def integrated_f(self, x_min, x_max, f, step):
        s = 0
        F_arr = [0]
        N = (x_max - x_min) / step
        N = math.ceil(N)
        step = (x_max - x_min) / N
        x1 = x_min
        x2 = x_min + step
        for i in range(N):
            s += (f(x1) + f(x2)) * step / 2
            F_arr.append(s)
            x1 = x2
            x2 += step

        F_max = F_arr[N]

        def i_f(x):
            if x <= x_min + 10**-13:
                return 0
            if x >= x_max - 10**-13:
                return F_arr[N]
            i = (x - x_min) / step
            i = math.floor(i)
            x1 = x_min + i*step
            # x2 = x1 + step
            return F_arr[i] + (F_arr[i+1] - F_arr[i]) * (x - x1) / step

        x_arr = [x_min]
        y_step = F_max / N
        j = 0
        for i in range(N):
            y_new = (i+1)*y_step
            while (F_arr[j+1] < y_new):
                j += 1
            x_old = x_min + step*j
            y_old = i_f(x_old)
            x_new = x_old + (y_new - y_old)*(step/(F_arr[j+1] - F_arr[j]))
            x_arr.append(x_new)

        def backward_i_f(y):

            if (y < -10**-13) or (y > F_max+10**-13):
                return math.nan
            if (abs(y) < 10**-13):
                return x_min
            if (abs(y - F_max) < 10**-13):
                return x_max

            i = math.floor(y/y_step)
            return x_arr[i] + (y-i*y_step)*((x_arr[i+1] - x_arr[i])/y_step)

        return i_f, backward_i_f, F_max

    def backward_psi_k(self, y, eps=-1):
        if eps <= 0:
            eps = self.backward_eps

        whole_part = math.floor(y)
        y -= whole_part
        return self.backward_psi_k_impl(y, eps) + whole_part

    def backward_psi_k_impl(self, y, eps=-1):
        if eps <= 0:
            eps = self.g**-(self.k+2)
        
        if y > 1-10**-13:
            return 1
        i = math.floor(y/self.y_step)
        j1 = self.interval_index_for_y[i]
        j2 = self.interval_index_for_y[i+1]+1
        while (j2 - j1 > 1):
            j_ = (j2 + j1) // 2
            if (self.interp_list[j_][1] >= y):
                j2 = j_
            else:
                j1 = j_

        if (j1 % 2 == 0):
            x0, y0 = self.interp_list[j1]
            return x0 + (y-y0)/self.B
        else:
            # print(y)
            x8, y8 = self.interp_list[j1]
            x10, y10 = self.interp_list[j2]
            # вызываем, чтобы гарантированно вычислить коэффициенты
            self.interpolated_psi_k_prime2((x8 + x10)/2)
            a, b, A, C = self.dict_param[(x8, x10)]
            x8_ = b-a
            x10_ = b+a
            y8_ = y8+(x8_-x8)*self.B
            y10_ = y10-(x10-x10_)*self.B
            if y < y8_ - 10**-13:
                # на нижнем линейном участке
                return x8 + (y8_ - y8)/self.B
            elif y > y10_ + 10**-13:
                # на верхнем линейном участке
                return x10 - (y8 - y8_)/self.B
            else:
                # на сигмоиде
                return x8_ + self.backward_ihat(y-y8_, a, A, eps)

    def backward_ihat_wo_b(self, eta, a, A):
        eta1 = eta/(A*a)
        ksi1 = 1 + self.back_i_stand_hat(eta1)
        ksi = ksi1 * a
        return ksi

    def backward_ihat(self, eta, a, A, eps):
        ksi_0 = 0
        if eta <= A*a*self.stand_norm:
            ksi_0 = self.backward_ihat_wo_b(eta, a, A)
        else:
            ksi_0 = 2*a
        b = a
        C = self.B*a + A*a*self.stand_norm/2

        def F(ksi):
            return self.ihat(ksi, A, a, self.B, b, C)

        def f(ksi):
            return self.hat_with_all_agrs(ksi, A, a, self.B, b, C)

        ksi = ksi_0
        ksi_min = -1
        ksi_max = -1
        delta_ksi_old = 3*a
        delta_ksi = 2*a
        counter = 0
        ser_len = 4
        newton_works = True
        def dscrp(ksi):
            # return F(ksi) + self.B * ksi - eta 
            return F(ksi) - eta 

        while abs(dscrp(ksi)) > eps:
            ksi = (ksi + (eta-F(ksi))/f(ksi)) / (1 + (self.B/f(ksi)))
            ksi = max(0, ksi)
            ksi = min(2*a, ksi)
            # NOTE: оцениваем, сходится ли Ньютон
            if counter % ser_len == 0:
                ksi_min = ksi
                ksi_max = ksi
            elif counter % ser_len == ser_len-1:
                delta_ksi_old = delta_ksi
                delta_ksi = ksi_max - ksi_min
                if delta_ksi >= delta_ksi_old - 10**-13:
                    newton_works = False
                    break
            else:
                ksi_min = min(ksi, ksi_min)
                ksi_max = max(ksi, ksi_max)
            counter += 1
        if newton_works:
            return ksi
        else:
            if dscrp(ksi_max) * dscrp(ksi_min) > 0:
                # print("не могу начать половинное деление")
                raise Exception ("не могу начать половинное деление")
            while delta_ksi >= eps:
                # TODO: метод половинного деления вместо Ньютона
                ksi = (ksi_max + ksi_min) / 2
                delta_ksi /= 2
                if dscrp(ksi_min) * dscrp(ksi) < 0:
                    ksi_max = ksi
                else:
                    ksi_min = ksi
            return ksi


    def integral(self, x_min, x_max, f, step):
        s = 0
        N = (x_max - x_min) / step
        N = math.ceil(N)
        step = (x_max - x_min) / N
        x1 = x_min
        x2 = x_min + step
        for i in range(N):
            s += (f(x1) + f(x2))*step/2
            x1 = x2
            x2 += step
        return s

    # def integrated_hat(x_min, x_max, x, step):
        # return integrated_f(x_min, x_max, hat, step)

    # a = (dx1 + dx2) / 2
    # b = x8 + a
    def hat_wo_b(self, x, A, a, B, b):
        if abs(x-b) < a:
            return A * math.exp(-1 / (1 - ((x - b) / a)**2))
        return 0

    def hat_with_b(self, x, A, a, B, b):
        return self.hat_wo_b(x, A, a, B, b) + B

    def generate_hat(self, A, a, B, b):
        def hat(x):
            return self.hat_with_b(x, A, a, B, b)
        return hat

    def ihat(self, x, A, a, B, b, C):
        x8_ = b - a
        x10_ = b + a
        y8_ = C - a*B - (self.stand_norm * A * a) / 2
        y10_ = C + a*B + (self.stand_norm * A * a) / 2
        if x < x8_:
            return y8_ + (x - x8_) * B
        if x > x10_:
            return y10_ + (x - x10_) * B
        return y8_ + self.i_stand_hat((x - b) / a) * a * A + B * (x - b + a)

    # def backward_ihat(self, x, A, a, B, b, C):
    #     x8_ = b - a
    #     x10_ = b + a
    #     y8_ = C - a*B - (self.stand_norm * A * a) / 2
    #     y10_ = C + a*B + (self.stand_norm * A * a) / 2
    #     # to be continue

    def generate_ihat(self, A, a, B, b, C):
        def ihat1(x):
            return self.ihat(x, A, a, B, b, C)
        return ihat1

    def gen_ihat_discrep(self, delta_x, x8, y8, y9, y10, step):
        delta_y1 = y9 - y8
        delta_y2 = y10 - y9
        x9 = x8 + delta_x
        x10 = x9 + delta_x

        if delta_y2 < delta_y1:
            def ihat_discrep(ksi):
                # ksi -= (x8 + delta_x)
                delta_y2_ = delta_y2 - self.B*(delta_x - ksi)
                a = (delta_x + ksi) / 2
                b = x8 + a
                if abs(a) < 10**-10:
                    a = 10**-10
                A = ((delta_y1 + delta_y2_) - (delta_x + ksi) * self.B) / (a * self.stand_norm)
                C = y8 + (delta_y1 + delta_y2_) / 2
                return y9 - self.ihat(x8 + delta_x, A, a, self.B, b, C), a, b, A, C
            return ihat_discrep

        else:
            def ihat_discrep(ksi):
                delta_y1_ = delta_y1 - self.B*(delta_x - ksi)
                a = (delta_x + ksi) / 2
                b = x10 - a
                if abs(a) < 10**-10:
                    a = 10**-10
                A = ((delta_y1_ + delta_y2) - (delta_x + ksi) * self.B) / (a * self.stand_norm)
                C = y10 - (delta_y1_ + delta_y2) / 2
                # return y9 - self.ihat(x8 + delta_x, A, a, self.B, b, C), a, b, A, C
                delat_y2_new = self.ihat(x10, A, a, self.B, b, C) - self.ihat(x10 - delta_x, A, a, self.B, b, C)
                y9_new = y10 - delat_y2_new
                return y9 - y9_new, a, b, A, C
            return ihat_discrep

    def gen_ihd(self, delta_x, x8, y8, y9, y10, step):
        ihd = self.gen_ihat_discrep(delta_x, x8, y8, y9, y10, step)

        def ihd1(ksi):
            res, _, _, _, _ = ihd(ksi)
            return res
        return ihd1

    def deriv(self, discrep, x0):
        d1 = discrep(x0 - self.tau / 2)
        d2 = discrep(x0 + self.tau / 2)
        return (d2 - d1) / self.tau

    # def deriv_2(self, discrep, y0, tau=0.05):
    #     d_minus = self.discrep(y0 - tau)
    #     d_plus = self.discrep(y0 + tau)
    #     d_zero = self.discrep(y0)
    #     return (d_minus - 2*d_zero + d_plus) / tau**2

    def Newton_method(self, discrep, x, delta, eps):
        dd1 = dd2 = 0
        delta_1 = 0
        while (dd1 * dd2 >= 0):
            delta_1 += delta
            dd1 = discrep(x - delta_1)
            dd2 = discrep(x + delta_1)

        a = x - delta_1
        b = x + delta_1

        f = discrep(x)
        while (abs(f) >= eps):
            newt = True
            f_2 = self.deriv(discrep, x)
            if f_2 == 0:
                newt = False
            else:
                x -= f / f_2
            if x < a or x > b or newt == False:
                x = (a + b) / 2
                f_a = discrep(a)
                f_x = discrep(x)
                if f_a * f_x < 0:
                    b = x
                else:
                    a = x
                x = (a + b) / 2
            f = discrep(x)
        return x

    def Newton_diap(self, discrep, a, b, eps):
        x = (a + b) / 2
        f = discrep(x)
        while (abs(f) >= eps):
            newt = True
            f_2 = self.deriv(discrep, x)
            if f_2 == 0:
                newt = False
            else:
                x -= f / f_2
            if x < a or x > b or newt == False:
                x = (a + b) / 2
                f_a = discrep(a)
                f_x = discrep(x)
                if f_a * f_x < 0:
                    b = x
                else:
                    a = x
                x = (a + b) / 2
            f = discrep(x)
        return x

    def find_ksi(self, delta_x, x8, y8, y9, y10, step, eps):
        ihd = self.gen_ihd(delta_x, x8, y8, y9, y10, step)
        ksi = self.Newton_diap(ihd, 0, delta_x, eps)
        return ksi


# def const(x):
#     return 1


# def lin(x):
#     return 2*x


# def sq(x):
#     return x**2


def sq1(x):
    return 4 - x**2


# f = sq1
# x_min = -2
# x_max = 2
# s = 0
# step = 0.01
# F_arr = [0]
# N = (x_max - x_min) / step
# N = math.ceil(N)
# step = (x_max - x_min) / N
# x1 = x_min
# x2 = x_min + step
# for i in range(N):
#     s += (f(x1) + f(x2)) * step / 2
#     F_arr.append(s)
#     x1 = x2
#     x2 += step
# plt.plot(F_arr)
# plt.show()

# gen2 = Psi_generator(k=2)
# i_const = gen2.integrated_f(-2, 2, const, 0.01)
# test_ = [i_const(x*(10**-2)) for x in range(-300, 300)]
# plt.plot(test_)
# plt.show()

# gen2 = Psi_generator(k=2)
# hd_test = gen2.generate_ihat(1, 1, 0, 2)
# hat_test = [gen2.i_stand_hat(x*(10**-2)) for x in range(-200, 200)]
# hd_array = [hd_test(x*(10**-2)) for x in range(0, 400)]
# plt.plot(hd_array)
# plt.show()

# gen2 = Psi_generator(k=2)
# def test_discrep(x):
#     return 3 - sq1(x)

# res = gen2.Newton_method(test_discrep, 0.5, 0.01, 0.01)
# print(res)

# gen2 = Psi_generator(k=2)

# TODO: fix bug x8 = 0.08, x9 = 0.09, x10 = 0.2
# x8 = 0.08
# x9 = 0.09
# x10 = 0.1
# y8 = gen2.psi_k(x8)
# y9 = gen2.psi_k(x9)
# y10 = gen2.psi_k(x10)
# # x9 = (x8 + x10) / 2
# step = 0.01
# delta_x = 0.01

# step = gen2.g**-(gen2.k) * gen2.default_step
# ksi = gen2.find_ksi(gen2.delta_x, x8, y8, y9, y10, step, step*gen2.eps_factor)

# ksi = gen2.find_ksi(delta_x, x8, y8, y9, y10, step, 0.001)
# ihd = gen2.gen_ihat_discrep(delta_x, x8, y8, y9, y10, step)
# ksi0 = ihd(0)
# ksi1 = ihd(delta_x)
# print(ksi0) 
# print(ksi1)
# _, a, b, A, C = ihd(ksi)
# ihat = gen2.generate_ihat(A, a, gen2.B, b, C)
# print(ihat(x8), ihat(x9), ihat(x10))

# ksi = self.find_ksi(self.delta_x, x8, y8, y9, y10, step, 0.001)
# ihd = self.gen_ihat_discrep(self.delta_x, x8, y8, y9, y10, step)
# _, a, b, A, C = ihd(ksi)


# gen2 = Psi_generator(k=3)
# # ihat = gen2.generate_ihat(3, 2, 0.25, 0.25, 5)
# # print(gen2.interpolated_psi_k(0.07))
# # print(gen2.psi_k(0.08))
# print('here')
# start = 8000
# end = 11000
# g_ = 10 ** - 5
# array2 = [gen2.interpolated_psi_k(i*(g_)) for i in range(start, end)]
# array2_lin = [gen2.lin_interp_psi_k(i*(g_)) for i in range(start, end)]
# array22 = [gen2.psi_k(i*(g_)) for i in range(start, end)]
# # array222 = [gen2.interpolated_psi_k_prime(i*(g_)) * 0.01 for i in range(start, end)]
# # array2222 = [gen2.interpolated_psi_k_prime2(i*(g_)) * 0.0001 for i in range(start, end)]
# array_1 = [gen2.interpolated_psi_k_prime(i*(g_)) for i in range(start, end)]
# array_11 = [gen2.interpolated_psi_k_prime2(i*(g_)) * 0.01 for i in range(start, end)]
# array_1_lin = [gen2.lin_interpolated_psi_k_prime(i*(g_), 10**-3) for i in range(start, end)]
# array_11_lin = [gen2.lin_interpolated_psi_k_prime2(i*(g_), 10**-3) * 0.01 for i in range(start, end)]

# # array_back = [gen2.back_i_stand_hat(gen2.F_max*i*0.001) for i in range(0, 1001)]
# # array_back2 = [(i*0.001) for i in range(0, math.ceil(gen2.F_max*1100))]
# # print(gen2.F_max)
# # print(array_back2)
# # plt.plot([gen2.F_max*i*0.001 for i in range(0, 1001)], array_back, label='back to hat') # без наклона
# # plt.show()


# x_ax = [i*(g_) for i in range(start, end)]
# plt.plot(x_ax, array22, label='psi, k = 3')
# plt.plot(x_ax, array2, label='interp psi, k = 3')
# plt.plot(x_ax, array2_lin, label='linear interp psi, k = 3')
# plt.legend()
# plt.show()

# plt.plot(x_ax, array_1, label='interp psi prime, k = 3')
# plt.plot(x_ax, array_11, label='interp psi prime2, k = 3')
# plt.legend()
# plt.show()

# plt.plot(x_ax, array_1_lin, label='lin interp psi prime, k = 3, delta_x = 10**-3')
# plt.plot(x_ax, array_11_lin, label='lin interp psi prime2, k = 3, delta_x = 10**-3')
# plt.legend()
# plt.show()
