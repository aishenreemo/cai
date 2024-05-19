#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "network.h"
#include "matrix.h"

network_item_t samples[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 1,
};

#define ACTIVATION ACTIVATION_SIGMOID

int main() {
	srand(time(0));

	struct network_t network = network_new(ACTIVATION, 3, 2, 2, 1);
	struct network_t network_gradient = network_new(ACTIVATION, 3, 2, 2, 1);

	struct matrix_t training_input = matrix_from(samples + 0, 2, 4, 3);
	struct matrix_t training_output = matrix_from(samples + 2, 1, 4, 3);

	network_item_t learning_rate = 10.0;

	network_rand(&network);

	double cost = network_cost(&network, &training_input, &training_output);

	while (cost > 1.e-4) {
		network_backpropagate(
			&network,
			&network_gradient,
			&training_input,
			&training_output
		);

		network_learn(
			&network,
			&network_gradient,
			learning_rate
		);

		cost = network_cost(&network, &training_input, &training_output);
		fprintf(stderr, "%lf\n", cost);
	}

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			MATRIX_AT(NETWORK_INPUT(network), 0, 0) = (network_item_t) i;
			MATRIX_AT(NETWORK_INPUT(network), 1, 0) = (network_item_t) j;
			network_forward(&network);
			printf("%d ^ %d = %lf\n", i, j, MATRIX_AT(NETWORK_OUTPUT(network), 0, 0));
		}
	}

	network_free(&network);
	network_free(&network_gradient);

	return 0;
}
