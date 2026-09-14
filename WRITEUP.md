# Write-up: RNN & LSTM from Scratch, and Why RNNs Struggle with Long-Term Dependencies

## 1. Overview
This project develops a vanilla Recurrent Neural Network (RNN) cell and a Long Short-Term Memory (LSTM) cell entirely from scratch using NumPy. Both architectures include manually derived Backpropagation Through Time (BPTT). The main objective is to study the vanishing gradient problem experimentally: how much gradients decrease as they propagate through time, and why LSTMs behave differently from vanilla RNNs.

All implementations are located in src/. The tests/ directory contains numerical gradient checks and a forward-pass comparison with torch.nn.LSTMCell. The experimental results are generated using src/gradient_analysis.py and src/train.py.

## 2. The Long-Term Dependency Task
The project uses a synthetic signal-in-noise recall problem, implemented in src/data.py. At the first timestep, t=0, the input contains a single bit of information, x_0 ∈ {-1, +1}. Every following input, from x_1 to x_{T-1}, consists of independent Gaussian noise that has no relevance to the correct answer.

The network makes its prediction using the hidden state at the final timestep, T-1, and must classify the sign of the original input bit using a sigmoid output and binary cross-entropy loss. The sequence length is set to T=60, which satisfies the requirement of using at least 50 timesteps.

This task is designed to isolate the ability to preserve information over time. The network does not need to learn complex patterns or interpret a meaningful sequence. It only needs to retain one bit of information introduced at the beginning and use it at the end. The experiment also examines whether the gradient associated with that bit can survive propagation backward through 59 timesteps.

## 3. Vanilla RNN: Forward Pass and Manual BPTT

Forward Pass
The vanilla RNN implementation in src/rnn.py updates its hidden state at each timestep using:

a_t = x_t @ Wxh.T + h_{t-1} @ Whh.T + bh
h_t = tanh(a_t)

After processing the sequence, a linear readout is applied once to the final hidden state:

y = h_T @ Why.T + by

Backward Pass

Let dh_t represent the gradient of the loss with respect to the hidden state at timestep t.
At the final timestep, the gradient comes from the output layer:
dh_{T-1} = dY @ Why

At earlier timesteps, the gradient is passed backward from the future hidden state. The main backward equations are:

da_t     = dh_t * (1 - tanh(a_t)^2)
dWxh    += da_t.T @ x_t
dWhh    += da_t.T @ h_{t-1}
dbh     += sum(da_t, axis=0)
dh_{t-1} = da_t @ Whh

The backward procedure is implemented using a reversed loop over the sequence in VanillaRNN.backward.

The important feature of this process is that each backward step involves multiplication by the recurrent weight matrix Whh and the derivative of the tanh activation. Since the tanh derivative is at most 1 and becomes very small when the activation saturates near -1 or +1, the gradient can decrease rapidly as it travels backward.

Over many timesteps, the gradient is approximately:
dh_0 ≈ dh_{T-1} * product[t=1 to T-1](
    tanh'(a_t) * Whh
)

Therefore, the gradient is effectively the result of multiplying many Jacobian matrices together. Whether it shrinks or grows depends mainly on the spectral properties of Whh and the degree of saturation of the tanh units. This repeated multiplication explains why vanilla RNNs are susceptible to vanishing and exploding gradients.

Correctness Verification
The test file tests/test_gradcheck.py compares the analytically calculated gradients for all parameters with gradients obtained using central finite differences. The relative error is approximately 1e-9, confirming that the BPTT implementation is correct.

## 4. Demonstrating Vanishing Gradients
The script src/gradient_analysis.py performs one forward and backward pass on a freshly initialized, untrained network. It records the average norm of the hidden-state gradient, ||dL/dh_t||, at every timestep.

Using an untrained model helps separate the effect of the architecture from changes that could occur during training.

For the vanilla RNN, the recurrent weights are multiplied by 0.5 relative to Xavier initialization. This places the recurrent matrix in a regime where its effective spectral radius is below 1.

For T=60, the results are:

Model	Gradient at t=0	Gradient at t=59	Decay ratio
Vanilla RNN	3.2e-20	3.6e-3	1.1e17
LSTM	4.1e-3	3.9e-3	0.95

The vanilla RNN's gradient decreases by approximately 17 orders of magnitude over 60 timesteps, whereas the LSTM's gradient remains nearly constant.

The complete gradient curve is stored in results/vanishing_gradients.png. On a logarithmic scale, the RNN produces an almost straight line, showing geometric decay. The LSTM curve remains approximately flat.

Effect of Recurrent Weight Scaling
Vanishing gradients are not an unavoidable outcome for every RNN configuration. They depend strongly on the recurrent weight scale.

The experiment in results/rnn_scale_sweep.csv tests recurrent weight scaling factors of 0.5, 1.0, 1.5, and 2.0.

Weight scale	Ratio (last/first)	Behavior
0.5	1.1e17	Vanishing
1.0	~1.0	Approximately balanced
1.5	1.4e-2	Mildly exploding
2.0	1.2e-4	Strongly exploding

Vanishing and exploding gradients arise from the same underlying process: repeated multiplication by Jacobians whose magnitudes are below or above 1. The difference is whether the repeated product becomes extremely small or extremely large.

This explains why vanilla RNNs are highly sensitive to initialization and hyperparameters. Only a limited range of settings provides numerically stable gradient propagation.

## 5. LSTM: Forward Pass and Manual Backward Pass

Forward Pass
The LSTM implementation in src/lstm.py combines four gates into a single linear transformation. The gates are arranged in the order (i, f, g, o) to match the layout used by nn.LSTMCell.

At each timestep:
z_t = x_t @ Wih.T + h_{t-1} @ Whh.T + b

i_t = sigmoid(z_i)
f_t = sigmoid(z_f)
g_t = tanh(z_g)
o_t = sigmoid(z_o)

c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
Here, i_t is the input gate, f_t is the forget gate, g_t is the candidate cell update, and o_t is the output gate.

The cell state c_t stores information across timesteps, while the hidden state h_t is generated from the cell state through the output gate.

Backward Pass
The LSTM backward equations are implemented in LSTM.backward.
The main derivatives are:
do      = dh_t * tanh(c_t)
dc_t   += dh_t_from_output * o_t * (1 - tanh(c_t)^2)

df      = dc_t * c_{t-1}
dc_{t-1} = dc_t * f_t

di      = dc_t * g_t
dg      = dc_t * i_t

The gate gradients are then multiplied by the derivatives of their respective activation functions. The resulting gradients are propagated through the stacked linear layer.

Correctness and Equivalence
The LSTM parameter gradients are checked using finite differences. The relative error is approximately 1e-7 or better for all five parameter tensors: Wih, Whh, b, Why, and by.

The project also includes tests/test_lstm_vs_pytorch.py, which compares the custom LSTM with nn.LSTMCell. It copies the PyTorch weights into the NumPy implementation and evaluates both models on identical inputs.

The test checks the hidden state and cell state at every timestep, with a tolerance of 1e-8. In the current environment, PyTorch is not installed, so the test reports that dependency is missing and skips the comparison gracefully. The test is ready to run when PyTorch is available.

## 6. Why the LSTM Avoids Vanishing Gradients: The Cell-State Highway
The main advantage of an LSTM comes from the way its cell state carries information and gradients across time.
Vanilla RNN

The RNN backward recurrence is:
dh_{t-1} = tanh'(a_t) * Whh.T * dh_t

At every timestep, the gradient is multiplied by both the recurrent matrix and the tanh derivative. If the spectral norm of Whh is below 1, the repeated product tends to shrink.

Consequently, the gradient accumulates a multiplicative penalty at every timestep, making long-range propagation difficult.

LSTM
The LSTM cell-state recurrence is:
dc_{t-1} = dc_t * f_t

This is the key difference. The cell-state gradient is multiplied by the forget gate's value rather than by the derivative of a saturating tanh activation.

If the forget gate remains close to 1, the gradient can travel through many timesteps with little attenuation:
1 * 1 * ... * 1 ≈ 1

The hidden state still passes through tanh and the output gate, but the cell-state pathway operates alongside that transformation. The cell state is updated additively:
c_t = f_t * c_{t-1} + i_t * g_t

Unlike the vanilla RNN hidden state, it is not repeatedly passed through a saturating nonlinearity at every timestep.

Forget-Gate Bias Experiment
The experiment in results/forget_bias_sweep.csv changes the initial forget-gate bias and measures the resulting gradient ratio.
Forget bias	Mean forget gate	Gradient ratio
0.0	0.50	9.5e11
1.0	0.73	1.7e4
2.0	0.88	2.1
3.0	0.95	0.95
5.0	0.99	0.48

As the forget gate approaches 1, the gradient decay ratio moves from severe vanishing to nearly lossless propagation.

This experiment demonstrates that the LSTM's advantage comes from its near-identity cell-state pathway. The gates help because they allow information and gradients to pass through a route that avoids repeated squashing transformations.

## 7. Training Comparison and Gradient Clipping
The script src/train.py trains three models on the same signal-recall task for 3000 SGD steps with sequence length T=60:
A vanilla RNN with recurrent weights scaled by 0.5.
The same vanilla RNN using global-norm gradient clipping with max_norm=1.0.
An LSTM with default initialization.
The held-out evaluation results are:
Model	Loss	Accuracy
Vanilla RNN	0.693 (≈ ln 2)	0.524
Vanilla RNN + gradient clipping	0.693	0.524
LSTM	0.0015	1.000

The vanilla RNN does not learn beyond chance-level performance. Its loss remains close to ln(2), which corresponds to a classifier predicting a probability of 0.5. 
Gradient clipping does not improve the result. It is not activated during any of the 3000 training steps because the raw gradient norm never reaches the clipping threshold.
This distinction is important: gradient clipping is intended to control exploding gradients, not restore gradients that have already vanished. In the vanishing regime, there is no large gradient to clip.

The LSTM, on the other hand, reaches almost perfect accuracy. Its training curve shows a noticeable improvement around step 1400, when it discovers the solution after an initial plateau.

## 8. Implementation Notes and Challenges
Several implementation details were important during development.
The LSTM backward pass requires handling two separate gradient contributions to the cell state: one coming from the hidden state through the output gate and tanh, and another carried backward from the next cell state through the forget gate. An early implementation accumulated the carried gradient one timestep out of phase. The finite-difference test detected the error, with approximately 100% relative error in the LSTM gradients.

The initial vanishing-gradient experiment used ordinary Xavier initialization for the RNN and did not show significant vanishing. This is expected because Xavier initialization aims to maintain activation and gradient variance near initialization. To demonstrate the problem clearly, the recurrent weight scale was deliberately changed from the balanced regime.

The unscaled Xavier-initialized RNN can learn this particular task at sequence lengths of T=100–150. This is important because vanishing gradients are not an absolute failure condition for every RNN. The severity of the problem depends on the initialization, weights, and task. This is why vanilla RNNs are often described as fragile and difficult to tune rather than universally incapable of learning long-term dependencies.

## 9. Conclusion

This project implements vanilla RNN and LSTM forward and backward passes from scratch using NumPy. Both models are verified through finite-difference gradient checks, and the LSTM implementation is additionally designed for comparison with nn.LSTMCell.
The experiments directly demonstrate the vanishing-gradient problem. Over 60 timesteps, the vanilla RNN's gradient decreases by approximately 17 orders of magnitude in the selected vanishing regime, while the LSTM maintains a nearly constant gradient.
The underlying mechanism is clear. In a vanilla RNN, the backward gradient repeatedly passes through the tanh derivative and recurrent weight matrix. In an LSTM, the cell state provides a near-identity path through the forget gate:

dc_{t-1} = dc_t * f_t

When the forget gate remains open, gradients can travel across many timesteps with little attenuation. The forget-bias experiment confirms this mechanism by showing a continuous transition from severe gradient decay to nearly lossless propagation.

Overall, the results show why LSTMs are better suited to learning long-term dependencies in situations where vanilla RNNs struggle with gradient propagation.

---
