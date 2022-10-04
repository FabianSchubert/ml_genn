import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn_eprop import EPropCompiler

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                log_latency_encode_data)

NUM_INPUT = 784
NUM_HIDDEN = 100
NUM_OUTPUT = 16
BATCH_SIZE = 128

TRAIN = True

labels = mnist.train_labels() if TRAIN else mnist.test_labels()
spikes = log_latency_encode_data(
    mnist.train_images() if TRAIN else mnist.test_images(),
    20.0, 51, 100)

network = SequentialNetwork()
with network:
    # Populations
    input = InputLayer(SpikeInput(max_spikes=BATCH_SIZE * calc_max_spikes(spikes)),
                                  NUM_INPUT)
    hidden = Layer(Dense(Normal(sd=1.0 / np.sqrt(NUM_INPUT))),
                   LeakyIntegrateFire(v_thresh=0.61, tau_mem=20.0,
                                      tau_refrac=5.0, 
                                      relative_reset=True,
                                      integrate_during_refrac=True),
                   NUM_HIDDEN)
    output = Layer(Dense(Normal(sd=1.0 / np.sqrt(NUM_HIDDEN))),
                   LeakyIntegrate(tau_mem=20.0, softmax=True, readout="sum_var"),
                   NUM_OUTPUT)

max_example_timesteps = int(np.ceil(calc_latest_spike_time(spikes)))
if TRAIN:
    compiler = EPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             optimiser="adam", batch_size=BATCH_SIZE)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = ["batch_progress_bar", "checkpoint"]
        metrics, _  = compiled_net.train({input: spikes},
                                         {output: labels},
                                         num_epochs=50, shuffle=True,
                                         callbacks=callbacks)
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")
else:
    # Load network state from final checkpoint
    network.load((15,))

    compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                 batch_size=BATCH_SIZE)
    compiled_net = compiler.compile(network)

    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        metrics, _  = compiled_net.evaluate({input: spikes},
                                            {output: labels})
        end_time = perf_counter()
        print(f"Accuracy = {100 * metrics[output].result}%")
        print(f"Time = {end_time - start_time}s")