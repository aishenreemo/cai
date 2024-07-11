#ifndef HISTORY_H
#define HISTORY_H

#include "network.h"
#include "matrix.h"

#define HISTORY_DEFAULT_PATH	"dist/sample.bin"
#define HISTORY_ENTRY_SIZE	128

struct history_t {
	int file;
	integer_t size;
	integer_t layer_count;
	integer_t neuron_count;
};

struct history_t history_new(char const *path, int flags);

void history_write_cfg(struct history_t *history, struct network_t *network);
void history_write_frame(struct history_t *history, struct network_t *network);

void history_close(struct history_t *history);

#endif // !HISTORY_H
