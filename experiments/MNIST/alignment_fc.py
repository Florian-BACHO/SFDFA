from pathlib import Path
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, "../../")  # Add repository root to python path

from Dataset import Dataset
from online_bats.Monitors import *
from online_bats.Layers import InputLayer, LIFLayer
from online_bats.Losses import *
from online_bats.Network import Network
from online_bats.Optimizers import *

PATH_TO_BATS = "../../../bats"

import sys

sys.path.insert(0, PATH_TO_BATS)  # Add repository root to python path

import bats
from bats import Layers as bats_layers
from bats import Losses as bats_losses

# Dataset
DATASET_PATH = Path("../../datasets/mnist.npz")

N_INPUTS = 28 * 28
SIMULATION_TIME = 0.2

# Hidden layer
N_NEURONS_1 = 800
TAU_S_1 = 0.130
THRESHOLD_HAT_1 = 0.2
DELTA_THRESHOLD_1 = 1 * THRESHOLD_HAT_1
SPIKE_BUFFER_SIZE_1 = 1

# Output_layer
N_OUTPUTS = 10
TAU_S_OUTPUT = 0.130
THRESHOLD_HAT_OUTPUT = 0.5
DELTA_THRESHOLD_OUTPUT = 1 * THRESHOLD_HAT_OUTPUT
SPIKE_BUFFER_SIZE_OUTPUT = 30

# Training parameters
N_TRAINING_EPOCHS = 100
N_TRAIN_SAMPLES = 60000
N_TEST_SAMPLES = 10000
TRAIN_BATCH_SIZE = 50
TRAIN_SAMPLE_REPEAT = 1
TEST_BATCH_SIZE = 100
N_TRAIN_BATCH = int(N_TRAIN_SAMPLES / TRAIN_BATCH_SIZE)
N_TEST_BATCH = int(N_TEST_SAMPLES / TEST_BATCH_SIZE)
TRAIN_PRINT_PERIOD = 0.1
TRAIN_PRINT_PERIOD_STEP = int(N_TRAIN_SAMPLES * TRAIN_PRINT_PERIOD / TRAIN_BATCH_SIZE)
TEST_PERIOD = 1.0  # Evaluate on test batch every TEST_PERIOD epochs
TEST_PERIOD_STEP = int(N_TRAIN_SAMPLES * TEST_PERIOD / TRAIN_BATCH_SIZE)
LEARNING_RATE = 5e-4
LR_DECAY_EPOCH = 1  # Perform decay very n epochs
LR_DECAY_FACTOR = 1.0
MIN_LEARNING_RATE = 0.0
FEEDBACK_LEARNING_RATE = 1e-4
REG_FACTOR = 0.00
TARGET_FALSE = 3
TARGET_TRUE = 15

# Plot parameters
EXPORT_METRICS = True
EXPORT_DIR = Path("./output_metrics")
SAVE_DIR = Path("./best_model")


def weight_initializer(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)


def weight_initializer_out(n_post: int, n_pre: int) -> cp.ndarray:
    return cp.random.uniform(-1.0, 1.0, size=(n_post, n_pre), dtype=cp.float32)

def export_weight_feedback_scatter(weights: cp.ndarray, feedbacks: cp.ndarray, path: Path):
    weights_flat = weights.flatten().get()
    feedbacks_flat = feedbacks.flatten().get()
    np.savez(path, weights=weights_flat, feedbacks=feedbacks_flat)

    plt.scatter(weights_flat, feedbacks_flat, alpha=0.3)
    plt.xlabel("Weight")
    plt.ylabel("Feedback")
    plt.savefig(path.with_suffix('.png'))
    plt.close()

if __name__ == "__main__":
    max_int = np.iinfo(np.int32).max
    np_seed = np.random.randint(low=0, high=max_int)
    cp_seed = np.random.randint(low=0, high=max_int)
    np.random.seed(np_seed)
    cp.random.seed(cp_seed)
    print(f"Numpy seed: {np_seed}, Cupy seed: {cp_seed}")

    if EXPORT_METRICS and not EXPORT_DIR.exists():
        EXPORT_DIR.mkdir()

    # Dataset
    print("Loading datasets...")
    dataset = Dataset(path=DATASET_PATH)

    print("Creating network...")
    network = Network()
    input_layer = InputLayer(n_neurons=N_INPUTS, name="Input layer")
    network.add_layer(input_layer, input=True)

    hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=N_NEURONS_1, n_outputs=N_OUTPUTS,
                            tau_s=TAU_S_1,
                            theta=THRESHOLD_HAT_1,
                            delta_theta=DELTA_THRESHOLD_1,
                            weight_initializer=weight_initializer,
                            max_n_spike=SPIKE_BUFFER_SIZE_1,
                            name="Hidden layer 1",
                            reg_factor=REG_FACTOR)
    network.add_layer(hidden_layer)

    output_layer = LIFLayer(previous_layer=hidden_layer, n_neurons=N_OUTPUTS, n_outputs=N_OUTPUTS,
                            tau_s=TAU_S_OUTPUT,
                            theta=THRESHOLD_HAT_OUTPUT,
                            delta_theta=DELTA_THRESHOLD_OUTPUT,
                            weight_initializer=weight_initializer_out,
                            max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                            name="Output layer",
                            is_output=True)
    network.add_layer(output_layer)

    loss_fct = SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)
    optimizer = AdamOptimizer(learning_rate=LEARNING_RATE)
    feedback_optimizer = AdamOptimizer(learning_rate=FEEDBACK_LEARNING_RATE)

    # BATS network
    bats_network = bats.Network()
    bats_input_layer = bats_layers.InputLayer(n_neurons=N_INPUTS, name="Input layer")
    bats_network.add_layer(bats_input_layer, input=True)

    bats_hidden_layer = bats_layers.LIFLayer(previous_layer=bats_input_layer, n_neurons=N_NEURONS_1,
                                             tau_s=TAU_S_1,
                                             theta=THRESHOLD_HAT_1,
                                             delta_theta=DELTA_THRESHOLD_1,
                                             weight_initializer=weight_initializer,
                                             max_n_spike=SPIKE_BUFFER_SIZE_1,
                                             name="Hidden layer 1")
    bats_network.add_layer(bats_hidden_layer)

    bats_output_layer = bats_layers.LIFLayer(previous_layer=bats_hidden_layer, n_neurons=N_OUTPUTS,
                                             tau_s=TAU_S_OUTPUT,
                                             theta=THRESHOLD_HAT_OUTPUT,
                                             delta_theta=DELTA_THRESHOLD_OUTPUT,
                                             weight_initializer=weight_initializer_out,
                                             max_n_spike=SPIKE_BUFFER_SIZE_OUTPUT,
                                             name="Output layer")
    bats_network.add_layer(bats_output_layer)

    bats_loss_fct = bats_losses.SpikeCountClassLoss(target_false=TARGET_FALSE, target_true=TARGET_TRUE)

    # Metrics
    training_steps = 0
    train_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_train")
    train_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_train")
    train_silent_label_monitor = SilentLabelsMonitor()
    train_time_monitor = TimeMonitor()
    train_alignment_monitor = AngleMonitor(hidden_layer.name, export_path=EXPORT_DIR / "alignment")
    train_monitors_manager = MonitorsManager([train_loss_monitor,
                                              train_accuracy_monitor,
                                              train_silent_label_monitor,
                                              train_time_monitor],
                                             print_prefix="Train | ")

    test_loss_monitor = LossMonitor(export_path=EXPORT_DIR / "loss_test")
    test_accuracy_monitor = AccuracyMonitor(export_path=EXPORT_DIR / "accuracy_test")
    test_learning_rate_monitor = ValueMonitor(name="Learning rate", decimal=5)
    # Only monitor LIF layers
    test_spike_counts_monitors = {l: SpikeCountMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
    test_silent_monitors = {l: SilentNeuronsMonitor(l.name) for l in network.layers if isinstance(l, LIFLayer)}
    test_norm_monitors = {l: WeightsNormMonitor(l.name, export_path=EXPORT_DIR / ("weight_norm_" + l.name))
                          for l in network.layers if isinstance(l, LIFLayer)}
    test_time_monitor = TimeMonitor()
    all_test_monitors = [test_loss_monitor, test_accuracy_monitor, train_alignment_monitor, test_learning_rate_monitor]
    all_test_monitors.extend(test_spike_counts_monitors.values())
    all_test_monitors.extend(test_silent_monitors.values())
    all_test_monitors.extend(test_norm_monitors.values())
    all_test_monitors.append(test_time_monitor)
    test_monitors_manager = MonitorsManager(all_test_monitors,
                                            print_prefix="Test | ")

    best_acc = 0.0
    print("Training...")
    for epoch in range(N_TRAINING_EPOCHS):
        train_time_monitor.start()
        dataset.shuffle()

        # Learning rate decay
        if epoch > 0 and epoch % LR_DECAY_EPOCH == 0:
            optimizer.learning_rate = np.maximum(LR_DECAY_FACTOR * optimizer.learning_rate, MIN_LEARNING_RATE)

        for batch_idx in range(N_TRAIN_BATCH):
            # Get next batch
            spikes, n_spikes, labels = dataset.get_train_batch(batch_idx, TRAIN_BATCH_SIZE, repeat=TRAIN_SAMPLE_REPEAT)

            # Inference
            network.reset()
            network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME, training=True)
            out_spikes, n_out_spikes, out_dir_derivatives = network.output_spike_trains

            # Predictions, loss and errors
            pred = loss_fct.predict(out_spikes, n_out_spikes)
            loss, loss_dir_derivative, output_errors = \
                loss_fct.compute_loss_and_directional_derivative(out_spikes, n_out_spikes, out_dir_derivatives, labels)

            pred_cpu = pred.get()
            loss_cpu = loss.get()
            n_out_spikes_cpu = n_out_spikes.get()

            # Update monitors
            train_loss_monitor.add(loss_cpu)
            train_accuracy_monitor.add(pred_cpu, labels)
            train_silent_label_monitor.add(n_out_spikes_cpu, labels)

            # Compute gradient
            avg_gradient = network.compute_avg_gradient(output_errors, n_out_spikes)
            avg_feedback_gradient = network.compute_avg_feedback_gradient(loss_dir_derivative)

            # Apply step
            deltas = optimizer.step(avg_gradient)
            feedback_deltas = feedback_optimizer.step(avg_feedback_gradient)

            bats_hidden_layer.weights = cp.copy(hidden_layer.weights)
            bats_output_layer.weights = cp.copy(output_layer.weights)

            network.apply_deltas(deltas)
            network.apply_feedback_deltas(feedback_deltas)
            del deltas

            # BATS Inference
            bats_network.reset()
            bats_network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME, training=True)
            bats_out_spikes, bats_n_out_spikes = bats_network.output_spike_trains

            # Predictions, loss and errors
            bats_pred = bats_loss_fct.predict(bats_out_spikes, bats_n_out_spikes)
            bats_loss, bats_errors = bats_loss_fct.compute_loss_and_errors(bats_out_spikes, bats_n_out_spikes,
                                                                           labels)

            # Compute gradient
            bats_gradient = bats_network.backward(bats_errors)
            bats_avg_gradient = [None if g is None else cp.mean(g, axis=0) for g in bats_gradient]

            train_alignment_monitor.add(avg_gradient[1].flatten().get(), bats_avg_gradient[1].flatten().get())

            training_steps += 1
            epoch_metrics = training_steps * TRAIN_BATCH_SIZE / N_TRAIN_SAMPLES
            # print(epoch_metrics)

            # Training metrics
            if training_steps % TRAIN_PRINT_PERIOD_STEP == 0:
                # Compute metrics

                train_monitors_manager.record(epoch_metrics)
                train_monitors_manager.print(epoch_metrics)
                train_monitors_manager.export()

            # Test evaluation
            if training_steps % TEST_PERIOD_STEP == 0:
                test_time_monitor.start()
                for batch_idx in range(N_TEST_BATCH):
                    spikes, n_spikes, labels = dataset.get_test_batch(batch_idx, TEST_BATCH_SIZE)
                    network.reset()
                    network.forward(spikes, n_spikes, max_simulation=SIMULATION_TIME)
                    out_spikes, n_out_spikes, _ = network.output_spike_trains

                    pred = loss_fct.predict(out_spikes, n_out_spikes)
                    loss = loss_fct.compute_loss(out_spikes, n_out_spikes, labels)

                    pred_cpu = pred.get()
                    loss_cpu = loss.get()
                    test_loss_monitor.add(loss_cpu)
                    test_accuracy_monitor.add(pred_cpu, labels)

                    for l, mon in test_spike_counts_monitors.items():
                        mon.add(l.spike_trains[1])

                    for l, mon in test_silent_monitors.items():
                        mon.add(l.spike_trains[1])

                for l, mon in test_norm_monitors.items():
                    mon.add(l.weights)

                test_learning_rate_monitor.add(optimizer.learning_rate)

                weights = output_layer.weights
                feedbacks = hidden_layer.feedback_weights.T

                records = test_monitors_manager.record(epoch_metrics)
                test_monitors_manager.print(epoch_metrics)
                test_monitors_manager.export()

                acc = records[test_accuracy_monitor]
                if acc > best_acc:
                    export_weight_feedback_scatter(weights, feedbacks, EXPORT_DIR / "weights_feedbacks_scatter")
                    best_acc = acc
                    network.store(SAVE_DIR)
                    print(f"Best accuracy: {np.around(best_acc, 2)}%, Networks save to: {SAVE_DIR}")
