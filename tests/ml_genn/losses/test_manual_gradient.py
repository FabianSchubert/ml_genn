import numpy as np
import pytest

from ml_genn import InputLayer, Layer, SequentialNetwork

from ml_genn.losses import ManualGradient

from ml_genn.compilers import EventPropCompiler

from ml_genn.neurons import IntegrateFireInput, LeakyIntegrate

from ml_genn.connectivity import Dense
from ml_genn.initializers import Uniform
from ml_genn.synapses import Exponential

N_IN = 5
N_OUT = 1

A = np.random.random((N_OUT, N_IN))

N_INPUT_FRAMES = 1000
INPUT_FRAME_TIMESTEPS = 1

N_EXAMPLES_TRAIN = 10000
N_EXAMPLES_TEST = 1000

x_train = np.random.random((N_EXAMPLES_TRAIN, N_IN))
y_train = x_train @ A.T
x_train = [np.repeat(x[np.newaxis, :], N_INPUT_FRAMES, axis=0) for x in x_train]
y_train = [np.repeat(y[np.newaxis, :], N_INPUT_FRAMES, axis=0) for y in y_train]

x_test = np.random.random((N_EXAMPLES_TEST, N_IN))
y_test = x_test @ A.T
x_test = [np.repeat(x[np.newaxis, :], N_INPUT_FRAMES, axis=0) for x in x_test]
y_test = [np.repeat(y[np.newaxis, :], N_INPUT_FRAMES, axis=0) for y in y_test]

NUM_EPOCHS = 10


def loss_func(y_pred, y_ture):
    return np.mean((y_pred - y_ture) ** 2)


def test_manual_gradient():
    network = SequentialNetwork()

    with network:
        inp = InputLayer(
            IntegrateFireInput(
                input_frames=N_INPUT_FRAMES, input_frame_timesteps=INPUT_FRAME_TIMESTEPS
            ),
            N_IN,
        )

        out = Layer(
            Dense(Uniform(min=-1.0, max=1.0), N_OUT),
            LeakyIntegrate(tau_mem=20.0, readout="var"),
            N_OUT,
            Exponential(5.0),
        )

    loss = ManualGradient(loss_func)

    compiler = EventPropCompiler(
        N_INPUT_FRAMES * INPUT_FRAME_TIMESTEPS,
        loss,
    )

    compiled_network = compiler.compile(network)

    import pdb

    pdb.set_trace()

    with compiled_network:
        metrics, _ = compiled_network.train(
            {inp: x_train},
            {out: y_train},
            num_epochs=NUM_EPOCHS,
            shuffle=True,
        )


if __name__ == "__main__":
    test_manual_gradient()
