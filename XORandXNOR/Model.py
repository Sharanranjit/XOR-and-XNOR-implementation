import numpy as np

# Transfer function
def sigmoid(x, Derivative=False):
    if not Derivative:
        return 1 / (1 + np.exp (-x))
    else:
        out = sigmoid(x)
        return out * (1 - out)
		
class NeuralNet:
        
    # Class Members describing the network
    
    numLayers = 0
    shape = None
    weights = []
    
    # Class Methods 
    
    def __init__(self, numNodes):
        
        # Layer info
        self.numLayers = len(numNodes) - 1
        self.shape = numNodes      

        # Input/Output data from last run
        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []
        
        # Create the weight arrays
        for (i,j) in np.column_stack((numNodes[:-1],numNodes[1:])):	#stacking numNodes[:-1] and numNodes[1:] gives all connections in network
            self.weights.append(np.random.normal(scale=0.1,size=(j,i+1)))
            self._previousWeightDelta.append(np.zeros((j,i+1)))       
        
    # Forward Pass method
    
    def FP(self, input):

        delta = []
        numExamples = input.shape[0]

        # Clean away the values from the previous layer
        self._layerInput = []
        self._layerOutput = []
        
        for index in range(self.numLayers):
            #Get input to the layer
            if index ==0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, numExamples])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,numExamples])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(sigmoid(layerInput))
            
        return self._layerOutput[-1].T
            
    # backPropagation method
    def backProp(self, input, target, learningRate = 0.2, momentum = 0.5):
        """Get the error, deltas and back propagate to update the weights"""

        delta = []
        numExamples = input.shape[0]
        
        # First run the network
        self.FP(input)
                 
        # Calculate the deltas for each node
        for index in reversed(range(self.numLayers)):
            if index == self.numLayers - 1:
                # If the output layer, then compare to the target values
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)	#L2 regression loss
                delta.append(output_delta * sigmoid(self._layerInput[index], True))
            else:
                # If a hidden layer. compare to the following layer's delta
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1,:] * sigmoid(self._layerInput[index], True))
                
        # Compute updates to each weight
        for index in range(self.numLayers):
            delta_index = self.numLayers - 1 - index
            
            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, numExamples])])
            else:
                layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1,self._layerOutput[index -1].shape[1]])])

            thisWeightDelta = np.sum(\
                    layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0) \
                    , axis = 0)
            
            weightDelta = learningRate * thisWeightDelta + momentum * self._previousWeightDelta[index]
            
            self.weights[index] -= weightDelta
            
            self._previousWeightDelta[index] = weightDelta
            
        return error