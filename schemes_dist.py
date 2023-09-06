from pennylane import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score
from ChemModel import translator, quantum_net
from Arguments import Arguments
import random
import pickle
import csv, os
import torch.multiprocessing as mp
import pennylane as qml
from math import pi

def get_time(f):
        def inner(*arg,**kwarg):
            s_time = time.time()
            res = f(*arg,**kwarg)
            e_time = time.time()
            print('Time：{}'.format(round((e_time - s_time), 2)))
            return res
        return inner

@get_time
def chemistry(hamiltonian, design, verbose=None):
    seeds = [20, 21, 30, 33, 36, 42, 43, 55, 67, 170]

    args = Arguments()
    lr = args.qlr

    symbols = ["O", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.45, -0.1525, -0.8454]])

# Building the molecular hamiltonian for the trihydrogen cation
    # hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=1)

    

    dev = qml.device("lightning.qubit", wires=args.n_qubits)
    @qml.qnode(dev, diff_method="adjoint")

    def cost_fn(theta):
        quantum_net(theta, design)
        return qml.expval(hamiltonian)
   
    energy = []
    for i in range(10):
        np.random.seed(seeds[i])
        if verbose: print('seed:', seeds[i])
        q_params = 2 * pi * np.random.rand(design['layer_repe'] * args.n_qubits * 2)
        opt = qml.GradientDescentOptimizer(stepsize = lr)
        # opt = qml.AdamOptimizer(stepsize=0.01, beta1=0.9, beta2=0.99, eps=1e-08)

        for n in range(100):
            q_params, prev_energy = opt.step_and_cost(cost_fn, q_params)
            if verbose: print(f"--- Step: {n}, Energy: {cost_fn(q_params):.8f}")
        energy.append(cost_fn(q_params))
    
    metrics = np.mean(energy)
    report = {'energy': metrics}
    print(metrics)
    return report

def search(hamiltonian, train_space, index, size):
    filename = 'train_results_{}.csv'.format(index)
    if os.path.isfile(filename) == False:
        with open(filename, 'w+', newline='') as res:
                writer = csv.writer(res)
                writer.writerow(['Num', 'sample_id', 'arch_code', 'Energy'])

    csv_reader = csv.reader(open(filename))
    i = len(list(csv_reader)) - 1
    j = index * size + i

    while len(train_space) > 0:
        net = train_space[i]
        print('Net', j, ":", net)
        design = translator(net)       
        report = chemistry(hamiltonian, design)       

        with open(filename, 'a+', newline='') as res:
            writer = csv.writer(res)           
            metrics = report['energy']
            writer.writerow([i, j, net, metrics])
        j += 1
        i += 1


if __name__ == '__main__':

    with open('hamiltonian', 'rb') as outfile:
        hamiltonian = pickle.load(outfile)
          
    train_space = []
    filename = 'data/train_space_1'

    with open(filename, 'rb') as file:
        train_space = pickle.load(file)

    num_processes = 10
    size = int(len(train_space) / num_processes)
    space = []
    for i in range(num_processes):
        space.append(train_space[i*size : (i+1)*size]) 
    
    with mp.Pool(processes = num_processes) as pool:
        pool.starmap(search, [(hamiltonian, space[i], i, size) for i in range(num_processes)])
    
       

    # net = [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 3, 7, 6, 3, 1, 1, 10, 6, 5, 6, 6, 7]
    # design = translator(net)   
    # report = chemistry(hamiltonian, design, 'print')
    