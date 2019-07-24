# Numpy Network to compute XOR and XNOR
## Model
`Model.py` Describes the network with forward and backward propagation used
## Train
`train.py` Describes the training process with 100000 epochs. Simply run `python train.py`.
After training, the weights are saved in `network.npy`
## Test
`test.py` For testing, the weights saved in `network.npy` is loaded. To test any two two bit binary numbers, put **num1** as first binary number and **num2** as second one.

For XOR, pass **ops** as 0

For XNOR, pass **ops** as 1

For example, run 
`python test.py --num1 1 0 --num2 0 1 --ops 0`

