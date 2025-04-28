# Neural Network for Even Parity Check

## a) The Solved Problem

The problem addressed in this experiment is the 4-bit even parity check problem. In this problem:

- The task is to determine whether a 4-bit binary input string contains an even or odd number of 1s.
- The desired output is 1 if the number of 1s is even (0, 2, or 4 ones), and 0 if odd (1 or 3 ones).
- This is a classic non-linearly separable problem, which means it cannot be solved by a single-layer perceptron.
- The parity problem serves as a benchmark for neural network capabilities as it requires the network to learn complex logical relationships.

There are 16 possible combinations of 4-bit inputs (2^4 = 16), and the neural network must correctly classify each of these patterns.

## b) The Used Method

### Neural Network Architecture

A three-layer feedforward neural network was implemented with the following structure:

- **Input Layer**: 5 neurons (4 for input bits + 1 bias neuron fixed at -1)
- **Hidden Layer**: Variable size (4, 6, 8, or 10 neurons, including 1 bias neuron fixed at -1)
- **Output Layer**: 1 neuron

### Training Algorithm

The network was trained using the backpropagation algorithm with the following parameters:

- Learning rate (η): 0.5
- Activation function: Sigmoid with λ = 1.0
- Error threshold: 0.001

### Training Process

1. Initialize weights with small random values between -0.5 and 0.5
2. Present all 16 input patterns to the network in each learning cycle
3. Calculate the output error for each pattern
4. Update weights using the backpropagation algorithm
5. Repeat until the total mean squared error falls below the desired threshold (0.001)

### Experimental Setup

Four separate experiments were conducted with different numbers of hidden neurons (4, 6, 8, and 10) to compare performance and find the optimal network configuration.

## c) Discussions on the Simulation Results

### Training Performance

| Hidden Neurons | Training Cycles | Final Error |
| -------------- | --------------- | ----------- |
| 4              | 155,058         | 0.001000    |
| 6              | 124,786         | 0.001000    |
| 8              | 25,924          | 0.001000    |
| 10             | 18,179          | 0.001000    |

### Key Findings

1. **Impact of Hidden Layer Size**:

   - A clear inverse relationship exists between the number of hidden neurons and the training time.
   - The 10-neuron network converged approximately 8.5 times faster than the 4-neuron network.
   - This demonstrates that having more computational units allows the network to find a solution path more efficiently.

2. **Learning Dynamics**:

   - All configurations exhibited a similar learning pattern: a long initial plateau where error remained near 2.0, followed by a sudden rapid decrease.
   - This suggests the network needed to reach a critical weight configuration before finding an effective solution path.
   - The long plateau phase indicates the difficulty of the parity problem for neural networks.

3. **Accuracy and Output Quality**:

   - All configurations eventually achieved 100% accuracy in classifying the 16 possible input patterns.
   - Output values:
     - 4 neurons: between 0.0106 and 0.9939
     - 6 neurons: between 0.0004 and 0.9960 (most well-separated)
     - 8 neurons: between 0.0030 and 0.9960
     - 10 neurons: between 0.0079 and 0.9922

4. **Computational Efficiency**:
   - The dramatic decrease in training cycles (155,058 → 18,179) when increasing hidden neurons indicates that the network's capacity to learn complex patterns improves significantly with more hidden units.
   - However, the relative improvement diminishes as more neurons are added (the improvement from 8 to 10 neurons is less significant than from 4 to 6 or 6 to 8).

### Conclusion

The simulation results demonstrate the effectiveness of the backpropagation algorithm in solving the non-linearly separable parity problem when given sufficient computational resources. The 8-neuron configuration appears to offer the optimal balance between network complexity and training efficiency for this problem.

The experiments confirm the theoretical understanding that more complex network architectures can learn faster but with diminishing returns. This has practical implications for neural network design: while adding more hidden neurons generally improves learning speed, there is a point beyond which additional complexity yields minimal benefits and may potentially lead to overfitting in other contexts.

For the 4-bit parity problem specifically, a network with 8 hidden neurons seems to be the most reasonable choice, providing excellent performance with moderate complexity.
