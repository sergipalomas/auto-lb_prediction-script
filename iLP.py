from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable


def solve_ilp(c1_n, c2_n):
    # Create the model
    model = LpProblem(name="opt-resources", sense=LpMaximize)
    # Initialize the decision variables
    nproc_x = LpVariable(name="nproc_x", lowBound=c1_n.nproc.min(), upBound=c1_n.nproc.max(), cat='Integer')
    nproc_y = LpVariable(name="nproc_y", lowBound=c2_n.nproc.min(), upBound=c2_n.nproc.max(), cat='Integer')

    # Define constraints
    #model += (c1_n.get_sypd(nproc_x) + c2_n.get_sypd(nproc_y) <= 1, "Same_SYPD_const")
    model += nproc_x - nproc_y <= 50

    # Define objective function (Maximize)
    #model += c1_n.get_fitness(nproc_x) + c2_n.get_fitness(nproc_y)

    model += nproc_x * 2 + nproc_y

    print(model)
    print('bye')