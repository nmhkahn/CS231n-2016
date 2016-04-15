import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

data = get_CIFAR10_data()
    
from cs231n.classifiers.convnet import *

log = open("log.txt", "w")

for i in xrange(10):
    lr = np.random.uniform(1e-2, 1e-5)
    model = ConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, filter_size=3, num_filters=(16, 16, 16, 16))

    print("lr: %e" % (lr))
    solver = Solver(model, data,
                    num_epochs=2, batch_size=50,
                    update_rule="adam",
                    optim_config={"learning_rate":lr},
                    verbose=True, print_every=50)

    solver.train()
    acc = solver.check_accuracy(solver.X_val, solver.y_val)
    log.write("lr: %e, acc: %f\n" % (lr, acc))

log.close()
