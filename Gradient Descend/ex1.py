import numpy as np
import matplotlib.pyplot as plt
import math

# coumpute the value of the function
def value(x):
    return x**2 + 5*np.sin(x)

# compute the gradient of the function
def grad(x):
    return 2*x + 5*np.cos(x)

def gradient_descend(eta, x0): # eta: learning rate, x0: initial point
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = gradient_descend(0.1, 5)
(x2, it2) = gradient_descend(0.1, -5)

print ("Solution x1 = %f, value = %f, obtained after %d iterations" %(x1[-1], value(x1[-1]), it1))
print ("Solution x2 = %f, value = %f, obtained after %d iterations" %(x2[-1], value(x2[-1]), it2))

'''
Solution x1 = -1.110341, value = -3.246394, obtained after 29 iterations
Solution x2 = -1.110667, value = -3.246394, obtained after 11 iterations
'''