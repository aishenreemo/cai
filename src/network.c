#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "network.h"
#include "history.h"
#include "matrix.h"

struct network_t network_new(integer_t layer_count, ...) {
	assert(layer_count > 1);

	struct network_t self;

	self.layer_count = layer_count;
	self.activation.function = activation_identity;
	self.activation.derivative = activation_identity_derivative;

	for (int i = 0; i < 2; i++) {
		unsigned long sizeof_matrix = sizeof(struct matrix_t);

		self.weights[i] = calloc(layer_count - 1, sizeof_matrix);
		self.biases[i] = calloc(layer_count - 1, sizeof_matrix);
		self.activations[i] = calloc(layer_count, sizeof_matrix);

		assert(self.weights[i] != NULL);
		assert(self.biases[i] != NULL);
		assert(self.activations[i] != NULL);
	}

	va_list args;
	va_start(args, layer_count);
	int input_count = va_arg(args, int);

	for (int i = 0; i < 2; i++) {
		self.activations[i][0] = matrix_new(input_count, 1);
	}

	for (int i = 0; i < layer_count - 1; i++) {
		int neuron_count = va_arg(args, int);

		for (int j = 0; j < 2; j++) {
			self.weights[j][i] = matrix_new(neuron_count, self.activations[j][i].cols);
			self.biases[j][i] = matrix_new(neuron_count, 1);
			self.activations[j][i + 1] = matrix_new(neuron_count, 1);
		}
	}

	va_end(args);

	return self;
}

struct network_t network_from(struct history_t *history) {
	struct network_t self;

	memcpy(&self.history, history, sizeof(struct history_t));

	enum history_token_t token = history_read_token(&self.history);

	assert(token == HISTORY_LC);
	self.layer_count = history_read_int(&self.history);

	token = history_read_token(&self.history);
	assert(token == HISTORY_A);

	network_set_activation(&self, history_read_int(&self.history));

	for (int i = 0; i < 2; i++) {
		unsigned long sizeof_matrix = sizeof(struct matrix_t);

		self.weights[i] = calloc(self.layer_count - 1, sizeof_matrix);
		self.biases[i] = calloc(self.layer_count - 1, sizeof_matrix);
		self.activations[i] = calloc(self.layer_count, sizeof_matrix);

		assert(self.weights[i] != NULL);
		assert(self.biases[i] != NULL);
		assert(self.activations[i] != NULL);
	}

	token = history_read_token(&self.history);
	assert(token == HISTORY_NC);
	integer_t input_count = history_read_int(history);

	for (int i = 0; i < 2; i++) {
		self.activations[i][0] = matrix_new(input_count, 1);
	}

	for (int i = 0; i < self.layer_count - 1; i++) {
		token = history_read_token(&self.history);
		assert(token == HISTORY_NC);
		integer_t neuron_count = history_read_int(history);

		for (int j = 0; j < 2; j++) {
			self.weights[j][i] = matrix_new(neuron_count, self.activations[j][i].cols);
			self.biases[j][i] = matrix_new(neuron_count, 1);
			self.activations[j][i + 1] = matrix_new(neuron_count, 1);
		}

	}

	token = history_read_token(&self.history);
	assert(token == HISTORY_N);

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

void network_init_write_history(struct network_t *network) {
	network->history = history_new("dist/sample.aiv", HISTORY_WRITE);

	history_add_token(&network->history, HISTORY_LC);
	history_add_int(&network->history, network->layer_count);
	history_add_token(&network->history, HISTORY_A);
	history_add_int(&network->history, network->activation.mode);

	for (int i = 0; i < network->layer_count; i++) {
		history_add_token(&network->history, HISTORY_NC);
		history_add_int(&network->history, network->activations[NETWORK_ORIGINAL][i].cols);
	}

	history_add_token(&network->history, HISTORY_N);
}

void network_end_write_history(struct network_t *network) {
	history_add_token(&network->history, HISTORY_EOF);
}

void network_write_frame(struct network_t *network) {
	history_add_token(&network->history, HISTORY_SPACE);
	for (int i = 0; i < network->layer_count - 1; i++) {
		history_add_token(&network->history, HISTORY_W);

		struct matrix_t *weights = network->weights[NETWORK_ORIGINAL] + i;
		int weights_count = weights->cols * weights->rows;

		for (int j = 0; j < weights_count; j++) {
		 	history_add_dec(&network->history, weights->items[j]);

			if (j == weights_count - 1) continue;
			history_add_token(&network->history, HISTORY_COMMA);
		}

		history_add_token(&network->history, HISTORY_B);
		struct matrix_t *biases = network->biases[NETWORK_ORIGINAL] + i;
		for (int j = 0; j < biases->cols; j++) {
		 	history_add_dec(&network->history, biases->items[j]);

			if (j == biases->cols - 1) continue;
			history_add_token(&network->history, HISTORY_COMMA);
		}
	}

	history_add_token(&network->history, HISTORY_N);
}

void network_read_frame(struct network_t *network, enum history_token_t *token) {
	for (int i = 0; i < network->layer_count - 1; i++) {
		*token = history_read_token(&network->history);
		assert(*token == HISTORY_W);

		struct matrix_t *weights = network->weights[NETWORK_ORIGINAL] + i;
		int weights_count = weights->cols * weights->rows;

		for (int j = 0; j < weights_count; j++) {
			weights->items[j] = history_read_dec(&network->history);

			if (j == weights_count - 1) continue;
			*token = history_read_token(&network->history);
			assert(*token == HISTORY_COMMA);
		}

		*token = history_read_token(&network->history);
		assert(*token == HISTORY_B);
		struct matrix_t *biases = network->biases[NETWORK_ORIGINAL] + i;
		for (int j = 0; j < biases->cols; j++) {
			biases->items[j] = history_read_dec(&network->history);

			if (j == biases->cols - 1) continue;
			*token = history_read_token(&network->history);
			assert(*token == HISTORY_COMMA);
		}
	}

	*token = history_read_token(&network->history);
	assert(*token == HISTORY_N);
}

void network_print(struct network_t *network) {
	printf("NN -> LC: %d, NC: ", network->layer_count);
	for (int i = 0; i < network->layer_count; i++) {
		printf("%d", network->activations[NETWORK_ORIGINAL][i].cols);
		if (i == network->layer_count - 1) continue;
		printf(", ");
	}

	printf("\n\t");
	for (int i = 0; i < network->layer_count - 1; i++) {
		printf("%d -> W: ", i);
		struct matrix_t *weights = network->weights[NETWORK_ORIGINAL] + i;
		int weights_count = weights->cols * weights->rows;
		for (int j = 0; j < weights_count; j++) {
			printf("%lf", weights->items[j]);

			if (j == weights_count - 1) continue;
			printf(", ");
		}

		printf(" B: ");
		struct matrix_t *biases = network->biases[NETWORK_ORIGINAL] + i;
		for (int j = 0; j < biases->cols; j++) {
			printf("%lf", biases->items[j]);

			if (j == biases->cols - 1) continue;
			printf(", ");
		}
		printf("\n\t");
	}
	printf("\n");
}

void network_free(struct network_t *network) {
	history_close(&network->history);

	for (int i = 0; i < 2; i++) {
		free(network->activations[i][0].items);

		for (int j = 0; j < network->layer_count - 1; j++) {
			free(network->weights[i][j].items);
			free(network->biases[i][j].items);
			free(network->activations[i][j + 1].items);
		}

		free(network->activations[i]);
		free(network->weights[i]);
		free(network->biases[i]);
	}
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
