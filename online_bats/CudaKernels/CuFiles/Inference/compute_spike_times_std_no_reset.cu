#define INFINITY __int_as_float(0x7f800000)

extern "C" {
    __device__ void get_sample_params(const int **spike_indices,
                                      const float **spike_times,
                                      const float **exp_tau_s,
                                      const float **exp_tau,
                                      const float **pre_dir_derivatives,
                                      const float **weights,
                                      const float **perturbations,
                                      int n_post_neurons, int n_pre_neurons, int sample_idx, int neuron_idx,
                                      int max_n_pre_spike) {
        int offset = sample_idx * max_n_pre_spike;

        *spike_indices += offset;
        *spike_times += offset;
        *exp_tau_s += offset;
        *exp_tau += offset;
        *pre_dir_derivatives += offset;

        *weights += neuron_idx * n_pre_neurons;
        *perturbations += sample_idx * n_post_neurons + neuron_idx;
    }

    __device__ void get_neuron_results(int **n_spikes,
                                       float **spike_times,
                                       float **post_dir_derivatives,
                                       float **trace_a,
                                       float **trace_b,
                                       float **eligibility_traces,
                                       int n_pre_neurons,
                                       int n_post_neurons,
                                       int sample_idx,
                                       int neuron_idx,
                                       int max_n_post_spike) {
        int sample_neuron_idx = (sample_idx * n_post_neurons + neuron_idx);
        int offset = sample_neuron_idx * max_n_post_spike;

        *n_spikes += sample_neuron_idx;
        *spike_times += offset;
        *post_dir_derivatives += offset;

        offset = sample_neuron_idx * n_pre_neurons;
        *trace_a += offset;
        *trace_b += offset;
        *eligibility_traces += offset;
    }

    __device__ bool compute_spikes(const float c,
                                   int *n_spikes,
                                   float *spike_times,
                                   float *post_dir_derivatives,
                                   float *eligibility_traces,
                                   float a,
                                   float *b,
                                   float b_hat,
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
                                   float perturbation) {
        float x, inside_log, inside_log_squared, tmp, factor, dt_dw, t_dir, a_scale, b_scale;

        // Compute until there is no spike anymore
        while (true) {
            // Compute discriminant
            tmp = (*b) * (*b) - 4 * a * c;

            if (tmp <= 0) // Negative discriminant, no spike --> stop
                return false;
            x = sqrtf(tmp);
            tmp = x + (*b);

            if (tmp == 0.0) // Division per zero, no spike --> stop
                return false;
            inside_log = 2.0 * a / tmp;

            //printf("%f\n", tmp);
            if (inside_log <= 0.0) // Negative log, no spike --> stop
                return false;

            tmp = tau * __logf(inside_log);

            // Spike time is before the last pre-spike or after the next spike --> stop
            if (tmp < last_spike || tmp > max_simulation || tmp > next_spike)
                return false;

            // Spike time is valid
            spike_times[*n_spikes] = tmp;
            last_spike = tmp;

            inside_log_squared = inside_log * inside_log;
            a_scale = a / inside_log_squared;
            b_scale = *b / inside_log;
            factor = tau / (a_scale);

            t_dir = factor * ((*dir_b) / inside_log - dir_a / inside_log_squared);
            post_dir_derivatives[*n_spikes] = t_dir + perturbation;

            *b = a / inside_log; // Apply reset to b
            *dir_b = (dir_a) / inside_log;

            // Update traces
            for (int i = 0; i < n_pre_neurons; i++) {
                if (trace_a[i] == 0.0)
                    continue;
                dt_dw = factor * (trace_b[i] / inside_log - trace_a[i] / inside_log_squared);
                eligibility_traces[i] += dt_dw;
            }
            (*n_spikes)++;
            if (*n_spikes >= max_n_post_spike) {
                return true;
            }
        }
    }

    __global__ void compute_spike_times_kernel(// Parameters
                                               const int *spike_indices,
                                               const float *spike_times,
                                               const float *exp_tau_s,
                                               const float *exp_tau,
                                               const float *pre_dir_derivatives,
                                               const float *weights,
                                               const float *perturbations,
                                               const float c,
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
                                               float *eligibility_traces) {
        int n_neurons = blockDim.x;
        int sample_idx = blockIdx.x;
        int neuron_idx = threadIdx.x;

        get_sample_params(&spike_indices, &spike_times, &exp_tau_s, &exp_tau, &pre_dir_derivatives, &weights,
                          &perturbations, n_neurons, n_pre_neurons, sample_idx, neuron_idx, max_n_pre_spike);
        get_neuron_results(&n_spikes, &out_spike_times, &post_dir_derivatives, &trace_a, &trace_b,
                           &eligibility_traces, n_pre_neurons, n_neurons, sample_idx, neuron_idx, max_n_post_spike);

        float a = 0.0;
        float b = 0.0;
        float b_hat = 0.0;
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
            //weight = (weights[pre_idx] < 0.0) ? -1.0 : 1.0;

            // Update a and traces
            tmp = weight * exp_tau_s[i];
            a += tmp;
            dir_a += tmp * pre_dir_derivatives[i];
            trace_a[pre_idx] += exp_tau_s[i];

            // Update b and traces
            tmp = weight * exp_tau[i];
            b += tmp;
            b_hat += tmp;
            dir_b += tmp * pre_dir_derivatives[i];
            trace_b[pre_idx] += exp_tau[i];

            next_i = i + 1;
            if (next_i < max_n_pre_spike)
                next_spike = spike_times[next_i];
            else
                next_spike = INFINITY;

            if (compute_spikes(c, n_spikes, out_spike_times, post_dir_derivatives, eligibility_traces,
                               a, &b, b_hat, trace_a, trace_b, dir_a, &dir_b, spike_times[i], next_spike,
                               tau, max_simulation, neuron_idx, n_pre_neurons,
                               max_n_post_spike, sample_idx, *perturbations))
                break; // Buffer full
        }
    }
}