#include <assert.h>
#include <stdint.h>

#include "network.h"
#include "history.h"

int main(void) {
	struct history_t history = history_new("dist/sample.aiv", HISTORY_READ);
	struct network_t network = network_from(&history);

	enum history_token_t token = history_read_token(&network.history);

	while (token != HISTORY_EOF) {
		network_read_frame(&network, &token);
		network_print(&network);

		token = history_read_token(&network.history);
	}

	network_free(&network);
}
