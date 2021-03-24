import math
import matplotlib.pyplot as plt
# import sys
# sys.setrecursionlimit(2000)


def func(x, y):
    return y**2/x


def solution(x, C):
    return -1/(math.log(x)+C)


def euclid(d1, d2):
    # return math.sqrt(d1 ** 2 + d2 ** 2)
    return (d1 ** 2 + d2 ** 2)


class Solver_RK4:
    def __init__(self, f, x0, x_left, x_right, y_left, y_right, step):
        self.f = f
        self.x0 = x0
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right
        self.step = step

    def step_forward(self, x, y, h):
        k1 = self.f(x, y)
        k2 = self.f(x + h/2, y + k1*h/2)
        k3 = self.f(x + h/2, y + k2*h/2)
        k4 = self.f(x + h, y + k3*h)
        return y + h*(k1 + 2*k2 + 2*k3 + k4)/6

    def left_step(self, x, y):
        return self.step_forward(x, y, -self.step)

    def right_step(self, x, y):
        return self.step_forward(x, y, self.step)

    def left_shot(self, x, y, need_values=False, values=[]):
        if x < self.x_left:
            return y, values
        if need_values:
            values.append(y)
        return self.left_shot(x-self.step, self.left_step(x, y), need_values, values)

    def right_shot(self, x, y, need_values=False, values=[]):
        if x > self.x_right:
            return y, values
        if need_values:
            values.append(y)
        return self.right_shot(x+self.step, self.right_step(x, y), need_values, values)

    def discrep(self, y0):
        y_right_res, _ = self.right_shot(self.x0, y0)
        y_left_res, _ = self.left_shot(self.x0, y0)
        right_discrep = self.y_right - y_right_res
        left_discrep = self.y_left - y_left_res
        return euclid(right_discrep, left_discrep)

    def deriv(self, y0, tau=0.05):
        d1 = self.discrep(y0 - tau / 2)
        d2 = self.discrep(y0 + tau / 2)
        return (d2 - d1) / tau

    def deriv_2(self, y0, tau=0.05):
        d_minus = self.discrep(y0 - tau)
        d_plus = self.discrep(y0 + tau)
        d_zero = self.discrep(y0)
        return (d_minus - 2*d_zero + d_plus) / tau**2

    def Newton_method(self, y, delta, eps):
        dd1 = dd2 = 0
        while (dd1 * dd2 >= 0):
            dd1 = self.deriv(y - delta)
            dd2 = self.deriv(y + delta)
            delta *= 2
        a = y - delta / 2
        b = y + delta / 2

        f = self.deriv(y)
        while (abs(f) >= eps):
            f_2 = self.deriv_2(y)
            y -= f / f_2
            if y < a:
                y = a
            if y > b:
                y = b
            f = self.deriv(y)
        return y

    def div_2(self, y0, delta, eps):
        '''метод половинного деления'''
        dd1 = dd2 = 0
        while (dd1 * dd2 >= 0):
            dd1 = self.deriv(y0 - delta)
            dd2 = self.deriv(y0 + delta)
            delta *= 2

        a = y0 - delta / 2
        b = y0 + delta / 2
        c = y0
        ddm = self.deriv(y0)

        while (abs(ddm) > eps):
            if dd1 * ddm < 0:
                b = c
            else:
                a = c

            c = (a + b) / 2
            dd1 = self.deriv(a)
            # dd2 = self.deriv(b)
            ddm = self.deriv(c)

        return c


x_array = [1 + 0.05*i for i in range(60)]
y_array = [solution(x, 1) for x in x_array]

sol = Solver_RK4(f=func, x0=2, x_left=1, x_right=3, y_left=-1, y_right=-0.5, step=0.05)
res = sol.div_2(-0.8, 0.1, 0.01)
res2 = sol.Newton_method(-0.8, 0.1, 0.01)

print(res)
print(res2)

y_left_res, left_array = sol.left_shot(2, -0.6, need_values=True, values=[])
y_right_res, right_array = sol.right_shot(2, -0.6, need_values=True, values=[])


left_array.reverse()
all_array = left_array+right_array[1:]
plt.plot(all_array)
# plt.plot(y_array)
# plt.plot(right_array)
# plt.show()
