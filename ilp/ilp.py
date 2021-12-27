from flask import Flask
import ghhops_server as hs

import rhino3dm
import numpy as np

from mip import Model, xsum, BINARY, INTEGER, minimize, maximize

app = Flask(__name__)
hops = hs.Hops(app)
    
    
@hops.component(
    "/solve",
    name="Solve",
    description="Solve the ILP Problem",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsNumber("Stock","S","Stock Lengths", hs.HopsParamAccess.LIST),
        hs.HopsNumber("PartLengths","P","Part Lengths", hs.HopsParamAccess.LIST),
        hs.HopsNumber("PartCounts","C","Part Counts", hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsNumber("Selection","S","Solved Result", hs.HopsParamAccess.LIST)
    ],
)
def solve(stock_lengths, part_lengths, part_requests):
    """
    Entry point function from hops
    """
    return solve_max(stock_lengths, part_lengths, part_requests)
    
def solve_default(stock_lengths, part_lengths, part_requests):
    """
    Strategy most similar to stock cutting example. 
    Built using loops and explicit utilization boolean variables
    Trends towards utilizing largest stock first
    """
    part_lengths = np.array(part_lengths)
    #part_lengths = np.array((8,7,5)).astype(int)
    part_count = len(part_lengths)

    part_requests = np.array(part_requests)
    #part_requests = np.array((3,2,1)).astype(int)

    stock_lengths = np.array(stock_lengths)
    #stock_lengths = np.array((15,14,10,11,18,21,16,13,14)).astype(int)
    stock_count = len(stock_lengths)
    
    max_parts = np.max(part_requests)
    
    #print(part_requests)
    #print(part_lengths)

    # Create Blank Model
    model = Model()
    model.max_mip_gap_abs = 1.5
    # Amount of each part used in that piece
    part_usage = {(i, j): model.add_var(obj=0, var_type=INTEGER, name="part_usage[%d,%d]" % (i, j), lb=0, ub=max_parts)
         for i in range(part_count) for j in range(stock_count)}
    # Whether the piece is used
    stock_usage = {j: model.add_var(obj=1, var_type=BINARY, name="stock_usage[%d]" % j)
         for j in range(stock_count)}

    # Constraints
    # Ensure enough parts are produced
    for i in range(part_count):
        model.add_constr(xsum(part_usage[i, j] for j in range(stock_count)) == part_requests[i])
    # Ensure the used amount of the bar is <= the usable amount of the bar (0 if unused)
    for j in range(stock_count):
        model.add_constr(xsum(part_lengths[i] * part_usage[i, j] for i in range(part_count)) <= stock_lengths[j] * stock_usage[j])
        

    # additional constraints to reduce symmetry
    # Put unused bars at end of list (reduces search space)
    #for j in range(1, stock_count):
    #    model.add_constr(stock_usage[j - 1] >= stock_usage[j])
        
    model.objective = minimize(xsum(stock_usage[i] for i in range(stock_count)))

    # optimizing the model
    model.optimize(max_nodes=10000)

    # printing the solution
    print('')
    print('Objective value: {model.objective_value:.3}'.format(**locals()))
    print('Solution: ', end='')
    for v in model.vars:
        if v.x > 1e-5:
            print('{v.name} = {v.x}'.format(**locals()))
            print('          ', end='')    
            
            
    output = [float(v.x) for v in model.vars]
    return output
    
    
def solve_waste(stock_lengths, part_lengths, part_requests):
    """
    Optimizes for minimizing waste from used piecess
    Does not attempt leftover usability
    """
    part_lengths = np.array(part_lengths)
    part_count = len(part_lengths)

    part_requests = np.array(part_requests)

    stock_lengths = np.array(stock_lengths)
    stock_count = len(stock_lengths)
    
    max_parts = np.max(part_requests)

    # Create Blank Model
    model = Model()
    model.max_mip_gap_abs = 1.5
    # Amount of each part used in that piece
    part_usage = {(i, j): model.add_var(obj=0, var_type=INTEGER, name="part_usage[%d,%d]" % (i, j), lb=0, ub=max_parts)
         for i in range(part_count) for j in range(stock_count)}
    # Whether the piece is used
    stock_usage = {j: model.add_var(obj=1, var_type=BINARY, name="stock_usage[%d]" % j)
         for j in range(stock_count)}

    # Constraints
    # Ensure enough parts are produced
    for i in range(part_count):
        model.add_constr(xsum(part_usage[i, j] for j in range(stock_count)) == part_requests[i])
    # Ensure the used amount of the bar is <= the usable amount of the bar (0 if unused)
    for j in range(stock_count):
        model.add_constr(xsum(part_lengths[i] * part_usage[i, j] for i in range(part_count)) <= stock_lengths[j] * stock_usage[j])
        
    model.objective = minimize(xsum((stock_lengths[j] * stock_usage[j]) - 
                                    (part_lengths[i] * part_usage[i, j]) 
                                    for i in range(part_count)
                                    for j in range(stock_count)))

    # optimizing the model
    model.optimize(max_nodes=10000)

    # printing the solution
    print('')
    print('Objective value: {model.objective_value:.3}'.format(**locals()))
    print('Solution: ', end='')
    for v in model.vars:
        if v.x > 1e-5:
            print('{v.name} = {v.x}'.format(**locals()))
            print('          ', end='')    
            
            
    output = [float(v.x) for v in model.vars]
    return output
    
    
def solve_max(stock_lengths, part_lengths, part_requests):
    """
    Ignores the utilized variable, tries to optimize the square of leftovers 
    Uses some SOS nonsense
    """
    part_lengths = np.array(part_lengths)
    part_count = len(part_lengths)

    part_requests = np.array(part_requests)

    stock_lengths = np.array(stock_lengths)
    stock_count = len(stock_lengths)
    
    max_parts = np.max(part_requests)
    largest_stock = np.max(stock_lengths)
    
    # Create Blank Model
    model = Model()
    model.max_mip_gap_abs = 1.5
    # Amount of each part used in that piece
    part_usage = {(i, j): model.add_var(obj=0, var_type=INTEGER, name="part_usage[%d,%d]" % (i, j), lb=0, ub=max_parts)
         for i in range(part_count) for j in range(stock_count)}


    # Constraints
    # Ensure enough parts are produced
    for i in range(part_count):
        model.add_constr(xsum(part_usage[i, j] for j in range(stock_count)) == part_requests[i])
    # Ensure the used amount of the bar is <= the usable amount of the bar
    for j in range(stock_count):
        model.add_constr(xsum(part_lengths[i] * part_usage[i, j] for i in range(part_count)) <= stock_lengths[j])
    
    # Create the nonlinear function for the objective
    score = [model.add_var(f"score_{i}") for i in range(stock_count)]
    for j in range(stock_count):
        d_count = 10
        v = [stock_lengths[j] * (v / (d_count - 1)) for v in range(d_count)] # X values for pow function
        vn = [pow(stock_lengths[j] - v[n],2) for n in range(d_count)]
        print(v)
        print(vn)
        w = [model.add_var(f"w_{j}_{v}") for v in range(d_count)]
        model.add_constr(xsum(w) == 1)
        
        model.add_constr(xsum((part_lengths[i] * part_usage[i, j])
                                                 for i in range(part_count))
                         ==
                         xsum(v[k] * w[k] for k in range(d_count)))
        model.add_constr(score[j] == xsum(vn[k] * w[k] for k in range(d_count)))
        model.add_sos([(w[k],v[k]) for k in range(d_count)],2)
    
    
        
    model.objective = maximize(xsum(score[i] for i in range(stock_count)))

    # optimizing the model
    status = model.optimize(max_nodes=10000)

    # printing the solution
    print(status)
    print('')
    print('Objective value: {model.objective_value:.3}'.format(**locals()))
    print('Solution: ', end='')
    for v in model.vars:
        if v.x > 1e-5:
            print('{v.name} = {v.x}'.format(**locals()))
            print('          ', end='')   

    output = [float(v.x) for v in model.vars]
    output = output[0:part_count * stock_count]
    return output


if __name__ == "__main__":
    app.run()