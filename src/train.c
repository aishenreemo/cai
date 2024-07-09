#include <stdbool.h>
#include <stdio.h>

#include "history.h"
#include "network.h"
#include "matrix.h"

#define USE_DIFFERENT_SEED	false
#define USE_UNLIMITED_LOOP	false

#define LOOP_LIMIT		5
#define COST_THRESHOLD		1.e-4

#ifdef USE_DIFFERENT_SEED
#include <stdlib.h>
#include <time.h>
#endif

decimal_t samples[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 0,
};

int main(void) {

#if USE_DIFFERENT_SEED
	srand(time(0));
#endif

	struct network_t network = network_new(3, 2, 2, 1);

	struct matrix_t training_input = matrix_from(samples + 0, 2, 4, 3);
	struct matrix_t training_output = matrix_from(samples + 2, 1, 4, 3);

	network_set_activation(&network, ACTIVATION_SIGMOID);
	network_init_write_history(&network);
	network_randomize(&network);

	decimal_t learning_rate = 10.0;
	decimal_t cost = network_cost(&network, &training_input, &training_output);

#if USE_UNLIMITED_LOOP
	while (cost > COST_THRESHOLD) {
#else
	for (int i = 0; i < LOOP_LIMIT; i++) {
#endif

		network_backpropagate(&network, &training_input, &training_output);
		network_learn(&network, learning_rate);
		network_write_frame(&network);
		network_print(&network);

		cost = network_cost(&network, &training_input, &training_output);
		// fprintf(stderr, "%lf\n", cost);
	}

	struct matrix_t output = network.activations[NETWORK_ORIGINAL][network.layer_count - 1];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			network_forward(&network, (decimal_t[2]) {i, j});
			printf("%d ^ %d = %lf\n", i, j, MATRIX_AT(output, 0, 0));
		}
	}

	network_end_write_history(&network);
	network_free(&network);

	return 0;
}
