#include <stdbool.h>
#include <fcntl.h>
#include <stdio.h>

#include "network.h"
#include "matrix.h"

#define USE_DIFFERENT_SEED	false
#define USE_UNLIMITED_LOOP	true
#define USE_HISTORY		true
#define USE_DEBUG		false

#define LOOP_LIMIT		1
#define COST_THRESHOLD		0.0001

#if USE_HISTORY
#include "history.h"
#endif

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

#if USE_HISTORY
	struct history_t history = history_new(HISTORY_DEFAULT_PATH, O_WRONLY | O_CREAT);
	history_write_cfg(&history, &network);
#endif

	struct matrix_t training_input = matrix_from(samples + 0, 2, 4, 3);
	struct matrix_t training_output = matrix_from(samples + 2, 1, 4, 3);

	network_set_activation(&network, ACTIVATION_SIGMOID);
	network_randomize(&network);

	decimal_t learning_rate = 10.0;

#if USE_UNLIMITED_LOOP
	decimal_t cost = network_cost(&network, &training_input, &training_output);

	while (cost > COST_THRESHOLD) {
#else // !USE_UNLIMITED_LOOP
	for (int i = 0; i < LOOP_LIMIT; i++) {
#endif // USE_UNLIMITED_LOOP

		network_backpropagate(&network, &training_input, &training_output);
		network_learn(&network, learning_rate);

#if USE_HISTORY
#if !USE_UNLIMITED_LOOP
		fprintf(stderr, "i: %d\n", i);
#endif // !USE_UNLIMITED_LOOP
		history_write_frame(&history, &network);
#endif // USE_HISTORY
       
#if USE_DEBUG
		network_print(&network);
#endif // USE_DEBUG

#if USE_UNLIMITED_LOOP
		cost = network_cost(&network, &training_input, &training_output);
#if USE_DEBUG
		fprintf(stderr, "\tcost: %lf\n", cost);
#endif // USE_DEBUG
#endif // USE_UNLIMITED_LOOP
	}

	struct matrix_t output = network.activations[NETWORK_ORIGINAL][network.layer_count - 1];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			network_forward(&network, (decimal_t[2]) {i, j});
			fprintf(stderr, "%d ^ %d = %lf\n", i, j, MATRIX_AT(output, 0, 0));
		}
	}

#if USE_HISTORY
	history_close(&history);
#endif

	network_free(&network);

	return 0;
}
