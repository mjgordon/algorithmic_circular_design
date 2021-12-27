from flask import Flask
import ghhops_server as hs

import rhino3dm
import numpy as np

from mip import Model, xsum, BINARY, INTEGER, minimize

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
    part_lengths = np.array(part_lengths)
    #part_lengths = np.array((8,7,5)).astype(int)
    part_count = len(part_lengths)

    part_requests = np.array(part_requests)
    #part_requests = np.array((3,2,1)).astype(int)

    stock_lengths = np.array(stock_lengths)
    #stock_lengths = np.array((15,14,10,11,18,21,16,13,14)).astype(int)
    stock_count = len(stock_lengths)
    
    max_parts = np.max(part_requests)
    
    print(part_requests)
    print(part_lengths)

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
    

if __name__ == "__main__":
    app.run()