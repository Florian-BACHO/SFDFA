extern "C" {
    __global__ void compute_pre_exps_kernel(const float *values,
                                            float *exp_tau_s,
                                            float *exp_tau,
                                            float tau_s,
                                            float tau,
                                            int n_neurons) {
        int max_n_spike = gridDim.z;

        int sample_idx = blockIdx.x;
        int pre_spike_idx = blockIdx.z;
        int block_neuron_idx = threadIdx.x;
        int neuron_block_idx = blockIdx.y;
        int tau_idx = threadIdx.y; // 0 if tau_s, 1 if tau

        int neuron_idx = neuron_block_idx * 512 + block_neuron_idx;
        if (neuron_idx >= n_neurons)
            return;

        // Global pre-spike index (i.e. relative to all samples' spikes)
        int spike_idx = ((sample_idx * n_neurons + neuron_idx) * max_n_spike) + pre_spike_idx;

        if (tau_idx == 0)
            exp_tau_s[spike_idx] = __expf(values[spike_idx] / tau_s);
        else
            exp_tau[spike_idx] = __expf(values[spike_idx] / tau);
    }
}