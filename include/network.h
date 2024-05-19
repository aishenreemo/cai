#ifndef NETWORK
#define NETWORK

#include <stdarg.h>
#include <assert.h>

#include "matrix.h"

#ifndef NETWORK_ITEM
#define NETWORK_ITEM matrix_item_t
typedef NETWORK_ITEM network_item_t;
#endif // NETWORK_ITEM

enum activation_t {
	ACTIVATION_SIGMOID,
	ACTIVATION_TANH,
	ACTIVATION_RELU
};

struct network_t {
	int count;
	enum activation_t method;
	struct matrix_t *weights;
	struct matrix_t *biases;
	struct matrix_t *activations;

	struct matrix_t training_input;
	struct matrix_t training_output;
};

struct network_t network_new(enum activation_t method, int count, ...);

void network_fill(struct network_t *dst, network_item_t value);
void network_rand(struct network_t *network);
void network_set_input(struct network_t *network, network_item_t *items);
void network_forward(struct network_t *network);

void network_activate(struct matrix_t *layer, enum activation_t method);

network_item_t network_activate_function(network_item_t x, enum activation_t method);
network_item_t network_activate_derivative(network_item_t x, enum activation_t method);

void network_print(struct network_t *const network);

void network_free(struct network_t *network);

void network_backpropagate(
	struct network_t *network,
	struct network_t *network_gradient,
	struct matrix_t *const training_input,
	struct matrix_t *const training_output
);

void network_learn(
	struct network_t *network,
	struct network_t *network_gradient,
	network_item_t learning_rate
);

network_item_t network_cost(
	struct network_t *network,
	struct matrix_t *const training_input,
	struct matrix_t *const training_output
);

#define NETWORK_INPUT(N) (N).activations[0]
#define NETWORK_OUTPUT(N) (N).activations[(N).count]

#endif // NETWORK
