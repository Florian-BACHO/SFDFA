#define INFINITY __int_as_float(0x7f800000)

extern "C" {
    __device__ void get_sample_params(const int **spike_indices,
                                      const float **spike_times,
                                      const float **exp_tau_s,
                                      const float **exp_tau,
                                      const float **pre_dir_derivatives,
                                      const float **weights,
                                      int n_post_neurons, int n_pre_neurons, int sample_idx, int neuron_idx,
                                      int max_n_pre_spike) {
        int offset = sample_idx * max_n_pre_spike;

        *spike_indices += offset;
        *spike_times += offset;
        *exp_tau_s += offset;
        *exp_tau += offset;
        *pre_dir_derivatives += offset;

        *weights += neuron_idx * n_pre_neurons;
    }

    __device__ void get_neuron_results(int **n_spikes,
                                       float **spike_times,
                                       float **post_dir_derivatives,
                                       float **trace_a,
                                       float **trace_b,
                                       float **last_dt_dw,
                                       float **eligibility_traces,
                                       const float **perturbations,
                                       int n_pre_neurons,
                                       int n_post_neurons,
                                       int sample_idx,
                                       int neuron_idx,
                                       int max_n_post_spike) {
        int sample_neuron_idx = (sample_idx * n_post_neurons + neuron_idx);
        int offset = sample_neuron_idx * max_n_post_spike;

        *n_spikes += sample_neuron_idx;
        *perturbations += sample_neuron_idx;
        *spike_times += offset;
        *post_dir_derivatives += offset;

        offset = sample_neuron_idx * n_pre_neurons;
        *trace_a += offset;
        *trace_b += offset;
        *last_dt_dw += offset;
        *eligibility_traces += offset;
    }

    __device__ bool compute_spikes(const float c,
                                   int *n_spikes,
                                   float *spike_times,
                                   float *post_dir_derivatives,
                                   float *last_dt_dw,
                                   float *eligibility_traces,
                                   const float *perturbations,
                                   float a,
                                   float *b,
                                   float *trace_a,
                                   float *trace_b,
                                   float dir_a,
                                   float *dir_b,
                                   float last_spike,
                                   float next_spike,
                                   float tau,
                                   float max_simulation,
                                   int neuron_idx,
                                   int n_pre_neurons,
                                   int max_n_post_spike,
                                   int sample_idx,
                                   bool is_output) {
        float tmp = __expf(last_spike / tau);
        if (*b / tmp < c * tau)
            return false;
        // Spike
        spike_times[*n_spikes] = last_spike;
        post_dir_derivatives[*n_spikes] = *dir_b + (*n_spikes == 0 ? *perturbations : 0.0);

        for (int i = 0; i < n_pre_neurons; i++) {
            if (trace_b[i] == 0.0)
                continue;
            eligibility_traces[i] -= trace_b[i] / tmp;
            trace_b[i] = 0.0;
        }
        *b = 0.0; // Apply reset to b
        *dir_b = 0.0;
        (*n_spikes)++;
        if (*n_spikes >= max_n_post_spike) {
            return true;
        }
        return false;
    }

    __global__ void compute_spike_times_kernel(// Parameters
                                               const int *spike_indices,
                                               const float *spike_times,
                                               const float *exp_tau_s,
                                               const float *exp_tau,
                                               const float *pre_dir_derivatives,
                                               const float *weights,
                                               const float *perturbations,
                                               bool is_output,
                                               float c,
                                               float tau,
                                               float max_simulation,
                                               int max_n_pre_spike,
                                               int max_n_post_spike,
                                               int n_pre_neurons,
                                               // Outputs
                                               int *n_spikes,
                                               float *out_spike_times,
                                               float *post_dir_derivatives,
                                               float *trace_a,
                                               float *trace_b,
                                               float *last_dt_dw,
                                               float *eligibility_traces) {
        int n_neurons = blockDim.x;
        int sample_idx = blockIdx.x;
        int neuron_idx = threadIdx.x;

        get_sample_params(&spike_indices, &spike_times, &exp_tau_s, &exp_tau, &pre_dir_derivatives, &weights,
                          n_neurons, n_pre_neurons, sample_idx, neuron_idx, max_n_pre_spike);
        get_neuron_results(&n_spikes, &out_spike_times, &post_dir_derivatives, &trace_a, &trace_b, &last_dt_dw,
                           &eligibility_traces, &perturbations, n_pre_neurons, n_neurons, sample_idx, neuron_idx,
                           max_n_post_spike);

        float a = 0.0;
        float b = 0.0;
        float dir_a = 0.0;
        float dir_b = 0.0;
        int pre_idx;
        float weight;
        int next_i;
        float next_spike;
        float tmp;

        for (int i = 0; i < max_n_pre_spike; i++) {
            if (spike_times[i] == INFINITY) // No spike anymore --> stop
                break;
            pre_idx = spike_indices[i];
            weight = weights[pre_idx];

            // Update b and traces
            tmp = weight * exp_tau[i];
            b += tmp;
            //dir_b += tmp / tau * pre_dir_derivatives[i];
            dir_b += weight * pre_dir_derivatives[i];
            trace_b[pre_idx] += exp_tau[i];

            next_i = i + 1;
            if (next_i < max_n_pre_spike)
                next_spike = spike_times[next_i];
            else
                next_spike = INFINITY;

            if (compute_spikes(c, n_spikes, out_spike_times, post_dir_derivatives, last_dt_dw, eligibility_traces,
                               perturbations, a, &b, trace_a, trace_b, dir_a, &dir_b, spike_times[i], next_spike,
                               tau, max_simulation, neuron_idx, n_pre_neurons,
                               max_n_post_spike, sample_idx, is_output))
                break; // Buffer full
        }
    }
}