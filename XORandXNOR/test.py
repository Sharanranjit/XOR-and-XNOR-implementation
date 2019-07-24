import argparse
import numpy as np
from Model import NeuralNet

parser = argparse.ArgumentParser(description='Computation of XOR or XNOR of two two bit binary numbers')
parser.add_argument('--num1', default=0, type=int, nargs='+', help='First two bit binary number')
parser.add_argument('--num2', default=0, type=int, nargs='+', help='Second two bit binary number')
parser.add_argument('--ops', default=0, type=int, help='XOR(0) or XNOR(1)')

args = parser.parse_args()

NN = NeuralNet((2,2,1))
NN.weights = np.load('network.npy',allow_pickle=True)
inp1 = args.num1
inp2 = args.num2
inp = np.column_stack((inp1,inp2))
result = np.squeeze(np.round(NN.FP(inp)).astype(int))
if args.ops == 0 :
	print(result)
else :
	print(1-result)