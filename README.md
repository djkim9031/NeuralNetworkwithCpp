# NeuralNetworkwithCpp
Simple neural network (Dense) and CNN architecture implemented with C++

A personal project to study the full low-level CNN architecture implementation.


It has two sub folders

1. CNN: access from main.cpp, full CNN implementation

2. Neural Network: access from main.cpp, full deep neural network (Dense) implemenation

3. [Newly included] Neural network implementation on ABB rapid program


## CNN with MNIST, printed outputs (examples):
<pre>
Start Training.....
Training model with 100 images

Epoch 1: [=========] -- Completed with the cost: 2.28876
Epoch 2: [=========] -- Completed with the cost: 2.19045
Epoch 3: [=========] -- Completed with the cost: 1.96059
Epoch 4: [=========] -- Completed with the cost: 1.67477
Epoch 5: [=========] -- Completed with the cost: 1.45272
Epoch 6: [=========] -- Completed with the cost: 1.2858
Epoch 7: [=========] -- Completed with the cost: 1.14447
Epoch 8: [=========] -- Completed with the cost: 1.02041
Epoch 9: [=========] -- Completed with the cost: 0.909825
Epoch 10: [=========] -- Completed with the cost: 0.813609

Training completed
Start testing on test dataset...
Correct label is: 7
Predicted label is: 7

--------------------
Correct label is: 2
Predicted label is: 0

--------------------
Correct label is: 1
Predicted label is: 1

--------------------
Correct label is: 0
Predicted label is: 0

--------------------
Correct label is: 4
Predicted label is: 4

--------------------
Correct label is: 1
Predicted label is: 1

--------------------
Correct label is: 4
Predicted label is: 4

--------------------
Correct label is: 9
Predicted label is: 4

--------------------
Correct label is: 5
Predicted label is: 6

--------------------
Correct label is: 9
Predicted label is: 7

--------------------
Correct label is: 0
Predicted label is: 0

--------------------
Correct label is: 6
Predicted label is: 6

--------------------
Correct label is: 9
Predicted label is: 7

--------------------
Correct label is: 0
Predicted label is: 0

--------------------
Correct label is: 1
Predicted label is: 1

--------------------
Correct label is: 5
Predicted label is: 3

--------------------
Correct label is: 9
Predicted label is: 7

--------------------
Correct label is: 7
Predicted label is: 7

--------------------
Correct label is: 3
Predicted label is: 3

--------------------
Correct label is: 4
Predicted label is: 4

--------------------
Program ended with exit code: 0

</pre>

## Neural network implementaion on ABB Rapid

<pre>
Pros-- 
1. As it is based on C, some optimizations are possible. 
(Training for 2 input feature, 2 hidden layers- each with 7 and 4 neurons- and 1 final layer with 1 neuron+sigmoid activation takes roughly a minute with 3 sample batches for 150 epochs)

2. Real-time monitoring of variables (such as weight gradients and weigh values at each layer) are possible. 
It's possible to constantly check if/where there is an vanishing/exploding gradients, and adjust the random distribution algorithm for weight initialization.

Cons--
1. Apparently, it is not object-oriented. Without being able to make a class, increasing a network's complexity is very difficult
2. Random value generation is impossible. Hence, in this code, a workaround method to generate a quasi-random uniform distribution (0,1) for weight initialization had to be implemented
3. Array dimension can only be up to 3. 4 dimensional arrays are systematically impossible to be implemented. Hence, a CNN with kernel filters cannot be implemented on ABB Rapid.
</pre>


