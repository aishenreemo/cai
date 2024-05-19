#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "network.h"
#include "matrix.h"

struct network_t network_new(enum activation_t method, int count, ...) {
	assert(count > 1);

	struct network_t network;

	network.count = count - 1;
	network.method = method;

	network.weights = calloc(network.count, sizeof(struct matrix_t));
	network.biases = calloc(network.count, sizeof(struct matrix_t));
	network.activations = calloc(count, sizeof(struct matrix_t));

	va_list args;
	va_start(args, count);

	network.activations[0] = matrix_new(va_arg(args, int), 1);

	for (int i = 0; i < network.count; i++) {
		int neuron_count = va_arg(args, int);
		network.weights[i] = matrix_new(neuron_count, network.activations[i].cols);
		network.biases[i] = matrix_new(neuron_count, 1);
		network.activations[i + 1] = matrix_new(neuron_count, 1);
	}

	va_end(args);

	return network;
}

void network_fill(struct network_t *network, network_item_t value) {
	matrix_fill(network->activations + 0, value);

	for (int i = 0; i < network->count; i++) {
		matrix_fill(network->weights + i, value);
		matrix_fill(network->biases + i, value);
		matrix_fill(network->activations + i + 1, value);
	}
}

void network_rand(struct network_t *network) {
	for (int i = 0; i < network->count; i++) {
		matrix_rand(network->weights + i);
		matrix_rand(network->biases + i);
	}
}

void network_set_input(struct network_t *network, network_item_t *items) {
	for (int i = 0; i < NETWORK_INPUT(*network).cols; i++) {
		MATRIX_AT(NETWORK_INPUT(*network), i, 0) = items[i];
	}
}

void network_forward(struct network_t *network) {
	for (int i = 0; i < network->count; i++) {
		matrix_mul(network->activations + i + 1, network->activations + i, network->weights + i);
		matrix_sum(network->activations + i + 1, network->biases + i);
		network_activate(network->activations + i + 1, network->method);
	}
}

void network_activate(struct matrix_t *layer, enum activation_t method) {
	assert(layer->rows == 1);

	for (int i = 0; i < layer->cols; i++) {
		MATRIX_AT(*layer, i, 0) = network_activate_function(MATRIX_AT(*layer, i, 0), method);
	}
}

network_item_t network_activate_function(network_item_t x, enum activation_t method) {
	switch (method) {
	case ACTIVATION_SIGMOID:
		return 1.0 / (1.0 + expf(-x));
	case ACTIVATION_TANH:
		return tanhf(x);
	case ACTIVATION_RELU:
		return (x < 0) ? 0 : x;
	}

	return 0.0;
}

network_item_t network_activate_derivative(network_item_t x, enum activation_t method) {
	switch (method) {
	case ACTIVATION_SIGMOID:
		return x * (1.0 - x);
	case ACTIVATION_TANH:
		return 1 - pow(x, 2);
	case ACTIVATION_RELU:
		return (x < 0) ? 0 : 1;
	}

	return 0.0;
}

void network_backpropagate(
	struct network_t *network,
	struct network_t *network_gradient,
	struct matrix_t *const training_input,
	struct matrix_t *const training_output
) {
	assert(network->count == network_gradient->count);
	assert(NETWORK_INPUT(*network).cols == training_input->cols);
	assert(NETWORK_OUTPUT(*network).cols == training_output->cols);
	assert(training_input->rows == training_output->rows);

	network_item_t sample_length = training_input->rows;

	network_fill(network_gradient, 0);

	// calculate gradients
	for (int i = 0; i < sample_length; i++) {
		network_item_t *inputs = &MATRIX_AT(*training_input, 0, i);

		network_set_input(network, inputs);
		network_forward(network);

		for (int j = 0; j <= network_gradient->count; j++) {
			matrix_fill(network_gradient->activations + j, 0);
		}

		for (int j = 0; j < training_output->cols; j++) {
			network_item_t predicted = NETWORK_OUTPUT(*network).items[j];
			network_item_t expected = MATRIX_AT(*training_output, j, i);

			NETWORK_OUTPUT(*network_gradient).items[j] = 2 * (predicted - expected);
		}

		for (int j = network_gradient->count; j > 0; j--) {
			struct matrix_t *layer = network->activations + j;
			struct matrix_t *layer_gradient = network_gradient->activations + j;

			struct matrix_t *bias_gradient = network_gradient->biases + (j - 1);
			struct matrix_t *weight_gradient = network_gradient->weights + (j - 1);

			struct matrix_t *prev_layer = network->activations + (j - 1);
			struct matrix_t *prev_weights = network->weights + (j - 1);
			struct matrix_t *prev_layer_gradient = network_gradient->activations + (j - 1);

			for (int k = 0; k < layer->cols; k++) {
				network_item_t g = MATRIX_AT(*layer_gradient, k, 0);

				network_item_t n = MATRIX_AT(*layer, k, 0);
				network_item_t d = network_activate_derivative(n, network->method);

				bias_gradient->items[k] += g * d;

				for (int l = 0; l < prev_layer->cols; l++) {
					network_item_t prev_n = MATRIX_AT(*prev_layer, l, 0);
					network_item_t w = MATRIX_AT(*prev_weights, k, l);

					MATRIX_AT(*weight_gradient, l, k) += g * d * prev_n;
					prev_layer_gradient->items[l] += g * d * w;
				}
			}
		}
	}

	// average of the gradients
	for (int i = 0; i < network_gradient->count; i++) {
		struct matrix_t *weights = network_gradient->weights + i;
		struct matrix_t *biases = network_gradient->biases + i;

		int weights_length = weights->rows * weights->cols;
		for (int j = 0; j < weights_length; j++) {
			weights->items[j] /= sample_length;
		}

		for (int j = 0; j < biases->cols; j++) {
			biases->items[j] /= sample_length;
		}
	}
}

void network_learn(
	struct network_t *network,
	struct network_t *network_gradient,
	network_item_t learning_rate
) {
	// apply gradients with learning_rate
	for (int i = 0; i < network_gradient->count; i++) {
		struct matrix_t *weights = network->weights + i;
		struct matrix_t *biases = network->biases + i;
		struct matrix_t *weights_gradient = network_gradient->weights + i;
		struct matrix_t *biases_gradient = network_gradient->biases + i;

		int weights_length = weights->rows * weights->cols;
		for (int j = 0; j < weights_length; j++) {
			int col = j % weights->cols;
			int row = j / weights->cols;

			network_item_t g = MATRIX_AT(*weights_gradient, col, row);
			MATRIX_AT(*weights, col, row) -= learning_rate * g;
		}

		for (int j = 0; j < biases->cols; j++) {
			network_item_t g = MATRIX_AT(*biases_gradient, j, 0);
			MATRIX_AT(*biases, j, 0) -= learning_rate * g;
		}
	}
}

network_item_t network_cost(
	struct network_t *network,
	struct matrix_t *const training_input,
	struct matrix_t *const training_output
) {
	assert(NETWORK_INPUT(*network).cols == training_input->cols);
	assert(NETWORK_OUTPUT(*network).cols == training_output->cols);
	assert(training_input->rows == training_output->rows);

	network_item_t cost = 0.0;
	for (int i = 0; i < training_input->rows; i++) {
		network_item_t *inputs = &MATRIX_AT(*training_input, 0, i);

		network_set_input(network, inputs);
		network_forward(network);

		for (int j = 0; j < training_output->cols; j++) {
			network_item_t a = MATRIX_AT(NETWORK_OUTPUT(*network), j, 0);
			network_item_t b = MATRIX_AT(*training_output, j, i);
			network_item_t d = a - b;
			cost += d * d;
		}
	}

	return cost / (training_input->rows * training_output->cols);
}

// I know you'll hate me for having to print every pointer here
// I don't know how my code segfaulting for using multiple of mallocs
//
// After hours of debugging, I've realized that the stdlib malloc()
// returns a pointer that is already allocated and i dont get why
//
// I'm paranoid now
//
// Anyway, calloc seems to work :D yeepee
void network_print(struct network_t *const network) {
	printf("NEURAL NETWORK\n");

	printf("--------------\n");
	printf("\033[1mWEIGHTS\033[0m\n");
	for (int i = 0; i < network->count; i++) {
		matrix_print(network->weights + i);
	}

	printf("--------------\n");
	printf("\033[1mBIASES\033[0m\n");
	for (int i = 0; i < network->count; i++) {
		matrix_print(network->biases + i);
	}

	printf("--------------\n");
	printf("\033[1mACTIVATIONS\033[0m\n");
	for (int i = 0; i < network->count + 1; i++) {
		matrix_print(network->activations + i);
	}

	printf("--------------\n\n");
}

void network_free(struct network_t *network) {
	free(network->activations[0].items);

	for (int i = 0; i < network->count; i++) {
		free(network->weights[i].items);
		free(network->biases[i].items);
		free(network->activations[i + 1].items);
	}
}
