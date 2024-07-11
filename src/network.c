#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "network.h"
#include "matrix.h"

static void network_alloc_matrices(struct network_t *network, integer_t layer_count) {
	network->matrices = calloc(layer_count * 6 - 4, sizeof(struct matrix_t));
	assert(network->matrices != NULL);

	for (int i = NETWORK_ORIGINAL; i <= NETWORK_GRADIENT; i++) {
		network->weights[i] = network->matrices + (i + 0) * (layer_count - 1);
		network->biases[i] = network->matrices + (i + 2) * (layer_count - 1);
		network->activations[i] = network->matrices + (i + 4) * (layer_count - 1);
	}

	network->activations[NETWORK_GRADIENT] += 1;
}

static void network_alloc_buffer(struct network_t *network, integer_t total_length) {
	network->buffer = calloc(total_length, sizeof(decimal_t));
	decimal_t *ptr = network->buffer + 0;
	network->matrices[0].items = ptr;
	for (int i = 1; i < network->layer_count * 6 - 4; i++) {
		ptr += network->matrices[i - 1].cols * network->matrices[i - 1].rows;
		network->matrices[i].items = ptr;
	}
}

struct network_t network_new(integer_t layer_count, ...) {
	assert(layer_count > 1);

	struct network_t self;

	self.layer_count = layer_count;
	self.activation.function = activation_identity;
	self.activation.derivative = activation_identity_derivative;

	network_alloc_matrices(&self, layer_count);

	va_list args;
	va_start(args, layer_count);
	integer_t total_length = 0;
	integer_t input_count = va_arg(args, integer_t);

	matrix_set_size(self.activations[NETWORK_ORIGINAL] + 0, input_count, 1);
	matrix_set_size(self.activations[NETWORK_GRADIENT] + 0, input_count, 1);

	for (int i = 0; i < layer_count - 1; i++) {
		integer_t neuron_count = va_arg(args, integer_t);
		integer_t previous_neuron_count = self.activations[NETWORK_ORIGINAL][i].cols;

		for (int j = 0; j < 2; j++) {
			matrix_set_size(self.weights[j] + i, neuron_count, previous_neuron_count);
			matrix_set_size(self.biases[j] + i, neuron_count, 1);
			matrix_set_size(self.activations[j] + i + 1, neuron_count, 1);
			total_length += neuron_count * previous_neuron_count + 2 * neuron_count;
		}
	}

	network_alloc_buffer(&self, total_length);

	va_end(args);

	return self;
}

void network_set_activation(struct network_t *network, enum activation_variant_t variant) {
	network->activation.mode = variant;
	switch (variant) {
		case ACTIVATION_IDENTITY:
			network->activation.function = activation_identity;
			network->activation.derivative = activation_identity_derivative;
			break;
		case ACTIVATION_SIGMOID:
			network->activation.function = activation_sigmoid;
			network->activation.derivative = activation_sigmoid_derivative;
			break;
		default:
			break;
	}
}

void network_randomize(struct network_t *network) {
	for (int i = 0; i < network->layer_count - 1; i++) {
		matrix_rand(network->weights[NETWORK_ORIGINAL] + i);
		matrix_rand(network->biases[NETWORK_ORIGINAL] + i);
	}
}

void network_reset_gradient(struct network_t *network) {
	matrix_fill(network->activations[NETWORK_GRADIENT] + 0, 0);

	for (int i = 0; i < network->layer_count - 1; i++) {
		matrix_fill(network->weights[NETWORK_GRADIENT] + i, 0);
		matrix_fill(network->biases[NETWORK_GRADIENT] + i, 0);
		matrix_fill(network->activations[NETWORK_GRADIENT] + i + 1, 0);
	}
}

void network_forward(struct network_t *network, decimal_t *items) {
	struct matrix_t *input = network->activations[NETWORK_ORIGINAL] + 0;
	for (int i = 0; i < input->cols; i++) {
		MATRIX_AT(*input, i, 0) = items[i];
	}

	for (int i = 0; i < network->layer_count - 1; i++) {
		struct matrix_t *activation_layer = network->activations[NETWORK_ORIGINAL] + i;
		struct matrix_t *weights = network->weights[NETWORK_ORIGINAL] + i;
		struct matrix_t *biases = network->biases[NETWORK_ORIGINAL] + i;

		matrix_mul(activation_layer + 1, activation_layer, weights);
		matrix_add(activation_layer + 1, activation_layer + 1, biases);

		network_activate(network, i + 1);
	}
}

void network_activate(struct network_t *network, integer_t layer_index) {
	struct matrix_t *layer = network->activations[NETWORK_ORIGINAL] + layer_index;
	assert(layer->rows == 1);

	for (int i = 0; i < layer->cols; i++) {
		MATRIX_AT(*layer, i, 0) = network->activation.function(MATRIX_AT(*layer, i, 0));
	}
}

void network_backpropagate(
	struct network_t *network,
	struct matrix_t *const training_input,
	struct matrix_t *const training_output
) {
	struct matrix_t *activations = network->activations[NETWORK_ORIGINAL];
	struct matrix_t *activationsg = network->activations[NETWORK_GRADIENT];
	struct matrix_t *input = activations + 0;
	struct matrix_t *output = activations + (network->layer_count - 1);
	struct matrix_t *outputg = activationsg + (network->layer_count - 1);

	assert(input->cols == training_input->cols);
	assert(output->cols == training_output->cols);
	assert(training_input->rows == training_output->rows);

	int sample_length = training_input->rows;

	network_reset_gradient(network);

	// calculate gradients
	for (int i = 0; i < sample_length; i++) {
		decimal_t *inputs = &MATRIX_AT(*training_input, 0, i);

		network_forward(network, inputs);
		for (int j = 0; j < network->layer_count; j++) {
			matrix_fill(activationsg + j, 0);
		}

		for (int j = 0; j < training_output->cols; j++) {
			decimal_t predicted = output->items[j];
			decimal_t expected = MATRIX_AT(*training_output, j, i);

			outputg->items[j] = 2 * (predicted - expected);
		}

		for (int j = network->layer_count - 1; j > 0; j--) {
			struct matrix_t *layer = activations + j;
			struct matrix_t *layerg = activationsg + j;

			struct matrix_t *biasg = network->biases[NETWORK_GRADIENT] + (j - 1);
			struct matrix_t *weightsg = network->weights[NETWORK_GRADIENT] + (j - 1);

			// ahem, i mean previous layer
			struct matrix_t *player = layer - 1;
			struct matrix_t *playerg = layerg - 1;
			struct matrix_t *pweights = network->weights[NETWORK_ORIGINAL] + (j - 1);

			for (int k = 0; k < layer->cols; k++) {
				decimal_t g = MATRIX_AT(*layerg, k, 0);

				decimal_t n = MATRIX_AT(*layer, k, 0);
				decimal_t d = network->activation.derivative(n);

				biasg->items[k] += g * d;

				for (int l = 0; l < player->cols; l++) {
					decimal_t prev_n = MATRIX_AT(*player, l, 0);
					decimal_t w = MATRIX_AT(*pweights, k, l);

					MATRIX_AT(*weightsg, l, k) += g * d * prev_n;
					playerg->items[l] += g * d * w;
				}
			}
		}
	}

	// average of the gradients
	for (int i = 0; i < network->layer_count - 1; i++) {
		struct matrix_t *weights = network->weights[NETWORK_GRADIENT] + i;
		struct matrix_t *biases = network->biases[NETWORK_GRADIENT] + i;

		int weights_length = weights->rows * weights->cols;
		for (int j = 0; j < weights_length; j++) {
			weights->items[j] /= sample_length;
		}

		for (int j = 0; j < biases->cols; j++) {
			biases->items[j] /= sample_length;
		}
	}
}

void network_learn(struct network_t *network, decimal_t learning_rate) {
	for (int i = 0; i < network->layer_count - 1; i++) {
		struct matrix_t *weights = network->weights[NETWORK_ORIGINAL] + i;
		struct matrix_t *biases = network->biases[NETWORK_ORIGINAL] + i;
		struct matrix_t *weights_gradient = network->weights[NETWORK_GRADIENT] + i;
		struct matrix_t *biases_gradient = network->biases[NETWORK_GRADIENT] + i;

		int weights_length = weights->rows * weights->cols;
		for (int j = 0; j < weights_length; j++) {
			int col = j % weights->cols;
			int row = j / weights->cols;

			decimal_t g = MATRIX_AT(*weights_gradient, col, row);
			MATRIX_AT(*weights, col, row) -= learning_rate * g;
		}

		for (int j = 0; j < biases->cols; j++) {
			decimal_t g = MATRIX_AT(*biases_gradient, j, 0);
			MATRIX_AT(*biases, j, 0) -= learning_rate * g;
		}
	}
}

void network_print(struct network_t *network) {
	fprintf(stderr, "NN -> LC: %d, NC: ", network->layer_count);
	for (int i = 0; i < network->layer_count; i++) {
		fprintf(stderr, "%d", network->activations[NETWORK_ORIGINAL][i].cols);
		if (i == network->layer_count - 1) continue;
		fprintf(stderr, ", ");
	}

	fprintf(stderr, "\n");
	for (int i = 0; i < network->layer_count - 1; i++) {
		fprintf(stderr, "\t%d -> W: ", i);
		struct matrix_t *weights = network->weights[NETWORK_ORIGINAL] + i;
		int weights_count = weights->cols * weights->rows;
		for (int j = 0; j < weights_count; j++) {
			fprintf(stderr, "%lf", weights->items[j]);

			if (j == weights_count - 1) continue;
			fprintf(stderr, ", ");
		}

		fprintf(stderr, " B: ");
		struct matrix_t *biases = network->biases[NETWORK_ORIGINAL] + i;
		for (int j = 0; j < biases->cols; j++) {
			fprintf(stderr, "%lf", biases->items[j]);

			if (j == biases->cols - 1) continue;
			fprintf(stderr, ", ");
		}
		fprintf(stderr, "\n");
	}
}

void network_free(struct network_t *network) {
	free(network->matrices);
	free(network->buffer);
}

decimal_t network_cost(
	struct network_t *network,
	struct matrix_t *const training_input,
	struct matrix_t *const training_output
) {
	struct matrix_t *network_input = network->activations[NETWORK_ORIGINAL] + 0;
	struct matrix_t *network_output =
		network->activations[NETWORK_ORIGINAL] +
		(network->layer_count - 1);

	assert(training_input->rows == training_output->rows);
	assert(network_input->cols == training_input->cols);
	assert(network_output->cols == training_output->cols);

	decimal_t cost = 0.0;
	for (int i = 0; i < training_input->rows; i++) {
		decimal_t *inputs = &MATRIX_AT(*training_input, 0, i);

		network_forward(network, inputs);

		for (int j = 0; j < training_output->cols; j++) {
			decimal_t a = MATRIX_AT(*network_output, j, 0);
			decimal_t b = MATRIX_AT(*training_output, j, i);
			decimal_t d = a - b;
			cost += d * d;
		}
	}

	return cost / (training_input->rows * training_output->cols);
}

decimal_t activation_identity(decimal_t x) {
	return x;
}

decimal_t activation_sigmoid(decimal_t x) {
	return 1.0 / (1.0 + exp(-x));
}

decimal_t activation_identity_derivative(decimal_t _) {
	return 1;
}

decimal_t activation_sigmoid_derivative(decimal_t x) {
	return x * (1.0 - x);
}
