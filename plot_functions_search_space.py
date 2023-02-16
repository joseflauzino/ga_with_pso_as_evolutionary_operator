import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 
from functions import *
from util import *

functions = ['sphere', 'rastrigin', 'ackley', 'eggholder', 'drop_wave']
for f in functions:
    f_name, f_, f_bounds, f_global_minimum = get_function_and_bounds(f)

    print('Ploting '+f_name+'...')

    X = np.linspace(f_bounds[0][0], f_bounds[0][1], 100)     
    Y = np.linspace(f_bounds[0][0], f_bounds[0][1], 100)   
    X, Y = np.meshgrid(X, Y) 

    Z = f_([X, Y])

    fig = plt.figure() 
    ax = fig.add_subplot(projection='3d') 
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    cmap=cm.nipy_spectral, linewidth=0.08,
    antialiased=True)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    plt.savefig('imgs/search_space/' + f_name + '.png')
print('Done!')