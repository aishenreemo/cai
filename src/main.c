#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "network.h"
#include "matrix.h"

decimal_t samples[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 0,
};

int main() {
	// srand(time(0));

	struct network_t network = network_new(3, 2, 2, 1);

	struct matrix_t training_input = matrix_from(samples + 0, 2, 4, 3);
	struct matrix_t training_output = matrix_from(samples + 2, 1, 4, 3);

	network_set_activation(&network, ACTIVATION_SIGMOID);
	network_randomize(&network);

	decimal_t learning_rate = 10.0;
	decimal_t cost = network_cost(&network, &training_input, &training_output);
#if 1
	while (cost > 1.e-4) {
#else
	for (int i = 0; i < 100; i++) {
#endif
		network_backpropagate(&network, &training_input, &training_output);
		network_learn(&network, learning_rate);

		cost = network_cost(&network, &training_input, &training_output);
		fprintf(stderr, "%lf\n", cost);
	}

	struct matrix_t output = network.activations[NETWORK_ORIGINAL][network.layer_count - 1];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			network_forward(&network, (decimal_t[2]) {i, j});
			printf("%d ^ %d = %lf\n", i, j, MATRIX_AT(output, 0, 0));
		}
	}

	network_free(&network);

	return 0;
}
