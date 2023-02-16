import numpy as np

def sphere(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total

def rastrigin(x):
    y = 10 * len(x) + sum(map(lambda i: i ** 2 -
                              10 * np.cos(2 * np.pi * i), x))
    return y

def ackley(x, a=20, b=0.2, c=2*np.pi):
    x = np.array(x)
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x*x) / d))
    cos_term = -np.exp(sum(np.cos(c*x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

def eggholder(x):
    x1 = x[0]
    x2 = x[1] if len(x) > 1 else 0
    a=np.sqrt(np.fabs(x2+x1/2+47))
    b=np.sqrt(np.fabs(x1-(x2+47)))
    c=-(x2+47)*np.sin(a)-x1*np.sin(b)
    return c

def drop_wave(x):
    x1 = x[0]
    x2 = x[1] if len(x) > 1 else 0
    b=0.5*(x1*x1+x2*x2)+2
    a=-(1+np.cos(12*np.sqrt(x1*x1+x2*x2)))/b
    return a