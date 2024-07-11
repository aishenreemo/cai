#include <assert.h>
#include <unistd.h>
#include <fcntl.h>

#include "history.h"
#include "matrix.h"
#include "network.h"

struct history_t history_new(char const *path, int flags) {
	struct history_t self;

	self.layer_count = 0;
	self.neuron_count = 0;
	self.size = 0;

	self.file = open(path, flags, 0644);
	assert(self.file != -1);

	return self;
}

void history_write_cfg(struct history_t *history, struct network_t *network) {
	history->layer_count = network->layer_count;
	write(history->file, &history->layer_count, sizeof(integer_t));

	for (int i = 0; i < history->layer_count; i++) {
		struct matrix_t *layer = network->activations[NETWORK_ORIGINAL] + i;

		integer_t layer_neuron_count = layer->cols;
		write(history->file, &layer_neuron_count, sizeof(integer_t));

		if (i + 1 == history->layer_count) continue;
		integer_t next_layer_neuron_count = (layer + 1)->cols;
		history->neuron_count += 
			2 * layer_neuron_count * next_layer_neuron_count +
			4 * layer_neuron_count;
	}

	fsync(history->file);
}

void history_write_frame(struct history_t *history, struct network_t *network) {
	for (int i = 0; i < history->neuron_count; i++) {
		decimal_t value = network->buffer[i];
		write(history->file, &value, sizeof(decimal_t));
	}
	fsync(history->file);
}

void history_close(struct history_t *history) {
	close(history->file);
}
