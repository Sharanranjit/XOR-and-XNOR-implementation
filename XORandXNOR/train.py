import numpy as np
from Model import NeuralNet

Input = np.array([[0,0],[0,1],[1,0],[1,1]])
Target = np.array([[0.0],[1.0],[1.0],[0.0]])

maxiterations = 100000
minerror = 1e-05

NN = NeuralNet((2,2,1))

for i in range(maxiterations + 1):
    Error = NN.backProp(Input, Target)
    if i % 25000 == 0:
        print("Iteration {0}\tError: {1:0.6f}".format(i,Error))
    if Error <= minerror:
        print("Minimum error reached at iteration {0}".format(i))
        break

Output = NN.FP(Input)
print ('Input \tOutput \t\tTarget')
for i in range(Input.shape[0]):
    print ('{0}\t {1} \t{2}'.format(Input[i], Output[i], Target[i]))

np.save('network.npy',NN.weights)