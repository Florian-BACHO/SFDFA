#define INFINITY __int_as_float(0x7f800000)

extern "C" {
    __device__ void get_sample_params(const int **spike_indices,
                                      const float **spike_times,
                                      const float **exp_tau_s,
                                      const float **exp_tau,
                                      const float **pre_dir_derivatives,
                                      const float **weights,
                                      const float **perturbations,
                                      int n_neurons,
                                      int sample_idx,
                                      int max_n_pre_spike,
                                      int neuron_c_idx,
                                      int n_weight_per_filter) {
        int sample_start_idx = sample_idx * max_n_pre_spike;


        *spike_indices += sample_start_idx;
        *spike_times += sample_start_idx;
        *exp_tau_s += sample_start_idx;
        *exp_tau += sample_start_idx;
        *pre_dir_derivatives += sample_start_idx;
        *weights += neuron_c_idx * n_weight_per_filter;
        *perturbations += neuron_c_idx;
    }

    __device__ void get_neuron_results(int **n_spikes,
                                       float **spike_times,
                                       float **post_dir_derivatives,
                                       float **trace_a,
                                       float **trace_b,
                                       float **eligibility_traces,
                                       int n_neurons,
                                       int sample_idx,
                                       int neuron_idx,
                                       int max_n_post_spike,
                                       int n_weight_per_filter) {
        int sample_neuron_idx = (sample_idx * n_neurons + neuron_idx);
        int res_start_idx = sample_neuron_idx * max_n_post_spike;

        *n_spikes += sample_neuron_idx;
        *spike_times += res_start_idx;
        *post_dir_derivatives += res_start_idx;

        res_start_idx = sample_neuron_idx * n_weight_per_filter;
        *trace_a += res_start_idx;
        *trace_b += res_start_idx;
        *eligibility_traces + res_start_idx;
    }

    __device__ bool compute_spikes(const float c,
                                   int *n_spikes,
                                   float *spike_times,
                                   float *post_dir_derivatives,
                                   float *eligibility_traces,
                                   float a,
                                   float *b,
                                   float *trace_a,
                                   float *trace_b,
                                   float dir_a,
                                   float *dir_b,
                                   float last_spike,
                                   float next_spike,
                                   float delta_theta_tau,
                                   float tau,
                                   float max_simulation,
                                   int neuron_idx,
                                   int max_n_post_spike,
                                   int n_weight_per_filter,
                                   int sample_idx,
                                   float perturbation) {
        float x, inside_log, tmp, dt_da, dt_db, dt_dw, t_dir;

        // Compute until there is no spike anymore
        while (true) {
            tmp = (*b) * (*b) - 4 * a * c;

            if (tmp < 0) { // Negative square root, no spike --> stop
                return false;
            }
            x = sqrtf(tmp);
            tmp = x + (*b);

            if (tmp == 0.0) { // Division per zero, no spike --> stop
                return false;
            }
            inside_log = 2 * a / tmp;

            if (inside_log < 0) { // Negative log, no spike --> stop
                return false;
            }

            tmp = tau * __logf(inside_log);
            // Spike time is before the last pre-spike or after the next spike --> stop
            if (tmp <= last_spike || tmp > max_simulation || tmp > next_spike) {
                return false;
            }

            // Spike time is valid
            dt_da = tau / a * (1 + c / x * inside_log);
            dt_db = -tau / x;
            spike_times[*n_spikes] = tmp;
            last_spike = tmp;
            t_dir = (dt_da * dir_a + dt_db * (*dir_b));
            post_dir_derivatives[*n_spikes] = t_dir + perturbation;

            // Update traces
            for (int i = 0; i < n_weight_per_filter; i++) {
                dt_dw = dt_da * trace_a[i] + dt_db * trace_b[i];
                eligibility_traces[i] += dt_dw;
                trace_b[i] = (trace_a[i] - dt_dw * a / tau) / inside_log;
            }
            *b = a / inside_log; // Apply reset to b
            *dir_b = (dir_a - t_dir * a / tau) / inside_log;
            (*n_spikes)++;
            if (*n_spikes >= max_n_post_spike) {
                return true;
            }
        }
    }

   __device__ bool get_spike_weight_and_traces(const float *weights,
                                               float *trace_a,
                                               float *trace_b,
                                               const int *pre_shape,
                                               const int *neuron_idx_3d,
                                               const int *filters_shape,
                                               const int *n_neurons_3d,
                                               float *weight,
                                               float **synapse_trace_a,
                                               float **synapse_trace_b,
                                               int spike_idx) {
        int tmp = pre_shape[1] * pre_shape[2];
        int spike_x = spike_idx / tmp;
        int spike_w = spike_idx % tmp;
        int spike_y = spike_w / pre_shape[2];
        int spike_c = spike_w % pre_shape[2];

        if (spike_x < neuron_idx_3d[0] || spike_x >= (neuron_idx_3d[0] + filters_shape[1]) ||
            spike_y < neuron_idx_3d[1] || spike_y >= (neuron_idx_3d[1] + filters_shape[2]))
            return false;

        int pos_x = spike_x - neuron_idx_3d[0];
        int pos_y = spike_y - neuron_idx_3d[1];
        int weight_idx = (pos_x * filters_shape[2] + pos_y) * filters_shape[3] + spike_c;
        //printf("%d %d | %d %d | %f\n", neuron_idx_3d[0], neuron_idx_3d[1], pos_x, pos_y, weights[weight_idx]);
        *weight = weights[weight_idx];
        *synapse_trace_a = trace_a + weight_idx;
        *synapse_trace_b = trace_b + weight_idx;
        return true;
   }

    __global__ void compute_spike_times_conv_kernel(// Parameters
                                                    const int *spike_indices,
                                                    const float *spike_times,
                                                    const float *exp_tau_s,
                                                    const float *exp_tau,
                                                    const float *pre_dir_derivatives,
                                                    const float *weights,
                                                    const float *perturbations,
                                                    const int *pre_shape,
                                                    const int *post_shape,
                                                    const int *filters_shape,
                                                    int n_neurons,
                                                    const float c,
                                                    float delta_theta_tau,
                                                    float tau,
                                                    float max_simulation,
                                                    int max_n_pre_spike,
                                                    int max_n_post_spike,
                                                    // Outputs
                                                    int *n_spikes,
                                                    float *out_spike_times,
                                                    float *post_dir_derivatives,
                                                    float *trace_a,
                                                    float *trace_b,
                                                    float *eligibility_traces) {
        int sample_idx = threadIdx.x;
        int neuron_idx_3d[3] = {blockIdx.x, blockIdx.y, blockIdx.z};
        int neuron_idx = (blockIdx.x * post_shape[1] + blockIdx.y) * post_shape[2] + blockIdx.z;
        int n_weight_per_filter = filters_shape[1] * filters_shape[2] * filters_shape[3];

        get_sample_params(&spike_indices, &spike_times, &exp_tau_s, &exp_tau, &pre_dir_derivatives, &weights,
                          &perturbations, n_neurons, sample_idx, max_n_pre_spike, neuron_idx_3d[2],
                          n_weight_per_filter);
        get_neuron_results(&n_spikes, &out_spike_times, &post_dir_derivatives, &trace_a, &trace_b, &eligibility_traces,
                           n_neurons, sample_idx, neuron_idx, max_n_post_spike, n_weight_per_filter);

        float a = 0.0;
        float b = 0.0;
        float dir_a = 0.0;
        float dir_b = 0.0;
        int pre_idx;
        float weight;
        int next_i;
        float next_spike;
        float tmp;
        float *synapse_trace_a;
        float *synapse_trace_b;

        for (int i = 0; i < max_n_pre_spike; i++) {
            if (spike_times[i] == INFINITY) // No spike anymore --> stop
                break;
            pre_idx = spike_indices[i];
            if (get_spike_weight_and_traces(weights, trace_a, trace_b, pre_shape, neuron_idx_3d, filters_shape,
                                            post_shape, &weight, &synapse_trace_a, &synapse_trace_b, pre_idx)) {
                // Update a and traces
                tmp = weight * exp_tau_s[i];
                a += tmp;
                dir_a += tmp * pre_dir_derivatives[i] * 2.0 / tau;
                *synapse_trace_a += exp_tau_s[i];

                // Update b and traces
                tmp = weight * exp_tau[i];
                b += tmp;
                dir_b += tmp * pre_dir_derivatives[i] / tau;
                *synapse_trace_b += exp_tau[i];
            }

            next_i = i + 1;
            if (next_i < max_n_pre_spike)
                next_spike = spike_times[next_i];
            else
                next_spike = INFINITY;

            if (compute_spikes(c, n_spikes, out_spike_times, post_dir_derivatives, eligibility_traces,
                               a, &b, trace_a, trace_b, dir_a, &dir_b, spike_times[i], next_spike, delta_theta_tau, tau,
                               max_simulation, neuron_idx, max_n_post_spike, n_weight_per_filter, sample_idx,
                               *perturbations))
                break; // Buffer full
        }
    }
}