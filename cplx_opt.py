import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def obj_f(nproc):
    nproc1 = nproc[0]
    nproc2 = nproc[1]
    r = (nproc1*c1_n.get_fitness(nproc1) + nproc2*c2_n.get_fitness(nproc2)) / (nproc1 + nproc2)
    return -r

def obj_f2(nproc, c1_n, c2_n):
    nproc1 = nproc[0]
    nproc2 = nproc[1]
    r = np.vectorize(c1_n.get_fitness)(nproc1) + np.vectorize(c2_n.get_fitness)(nproc2)
    return r


def plot_obj_f(c1_n, c2_n, res):
    nproc1 = np.linspace(c1_n.nproc.min(), c1_n.nproc.max(), 50)
    nproc2 = np.linspace(c2_n.nproc.min(), c2_n.nproc.max(), 50)

    X, Y = np.meshgrid(nproc1, nproc2)
    Z = obj_f2([X, Y], c1_n, c2_n)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 1000)
    ax.set_xlabel(c1_n.name)
    ax.set_ylabel(c2_n.name)
    ax.set_zlabel('obj_f')
    ax.view_init(30, 30)
    ax.plot([res[0]], [res[1]], [res[2]+0.1], 'o', color='r')
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel(c1_n.name)
    ax.set_ylabel(c2_n.name)
    ax.set_zlabel('obj_f')
    ax.plot([res[0]], [res[1]], [res[2]], 'o', color='r')
    ax.view_init(30, 30)
    ax.set_title('Surface plot')
    plt.show()


def find_optimal(c1_n, c2_n):

    constraint1 = ({'type': 'ineq', 'fun': lambda nproc: - c1_n.get_sypd(nproc[0]) + c2_n.get_sypd(nproc[1]) + 0.5},
                   {'type': 'ineq', 'fun': lambda nproc: + c1_n.get_sypd(nproc[0]) - c2_n.get_sypd(nproc[1]) + 0.5})
    constraint1 = ({'type': 'ineq', 'fun': lambda nproc: - c1_n.get_sypd(nproc[0]) + c2_n.get_sypd(nproc[1]) + 4})

    objective = lambda nproc: -(c1_n.get_fitness(nproc[0]) + c2_n.get_fitness(nproc[1]))

    bounds_nproc1 = (c1_n.nproc.min(), c1_n.nproc.max())
    bounds_nproc2 = (c2_n.nproc.min(), c2_n.nproc.max())
    bounds = [bounds_nproc1, bounds_nproc2]
    ig = np.asarray([500, 500], dtype=np.float32)
    res = minimize(objective, ig, method='SLSQP', bounds=bounds, constraints=constraint1, options={'disp': True})
    nproc_c1 = round(res.x[0])
    nproc_c2 = round(res.x[1])
    print("optimal var: x1 = ", nproc_c1, " x2 = ", nproc_c2)

    print("Optimal solution:\n")
    print("%s:\n"
          "nproc: %i\n"
          "SYPD: %.2f\n"
          "Fitness: %.3f\n" % (c1_n.name, nproc_c1, c1_n.get_sypd(nproc_c1), c1_n.get_fitness(nproc_c1)))
    print("%s:\n"
          "nproc: %i\n"
          "SYPD: %.2f\n"
          "Fitness: %.3f\n" % (c2_n.name, nproc_c2, c2_n.get_sypd(nproc_c2), c2_n.get_fitness(nproc_c2)))
    result = (nproc_c1, nproc_c2, -objective([nproc_c1, nproc_c2]))
    plot_obj_f(c1_n, c2_n, result)