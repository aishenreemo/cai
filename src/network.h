#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"

#define NETWORK_ORIGINAL 0
#define NETWORK_GRADIENT 1

typedef decimal_t (*activation_t)(decimal_t);

enum activation_variant_t {
	ACTIVATION_IDENTITY,
	ACTIVATION_SIGMOID,
};

struct activation_t {
	enum activation_variant_t mode;
	decimal_t (*function)(decimal_t);
	decimal_t (*derivative)(decimal_t);
};

struct network_t {
	// total length of layers
	integer_t layer_count;
	// dynamically allocated storage for all numbers stored in the network
	decimal_t *buffer;
	// dynamically allocated storage for all matrices
	struct matrix_t *matrices;
	// references to matrices
	// matrix.items is a slice of buffer
	struct matrix_t *weights[2];
	struct matrix_t *biases[2];
	struct matrix_t *activations[2];
	// activation function which can be customized by user
	struct activation_t activation;
};

struct network_t network_new(uint32_t layer_count, ...);

void network_set_activation(struct network_t *network, enum activation_variant_t variant);

void network_randomize(struct network_t *network);
void network_reset_gradient(struct network_t *network);
void network_forward(struct network_t *network, decimal_t *items);
void network_activate(struct network_t *network, uint32_t layer_index);
void network_learn(struct network_t *network, decimal_t learning_rate);
void network_print(struct network_t *network);
void network_free(struct network_t *network);

void network_backpropagate(
	struct network_t *network,
	struct matrix_t *const training_input,
	struct matrix_t *const training_output
);

decimal_t network_cost(
	struct network_t *network,
	struct matrix_t *const training_input,
	struct matrix_t *const training_output
);

decimal_t activation_identity(decimal_t);
decimal_t activation_sigmoid(decimal_t);
decimal_t activation_identity_derivative(decimal_t);
decimal_t activation_sigmoid_derivative(decimal_t);

#endif // !NETWORK_H
