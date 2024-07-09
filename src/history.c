#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>

#include "history.h"
#include "matrix.h"

struct history_t history_new(char const *file_path, enum history_mode_t mode) {
	struct history_t self;

	self.file = fopen(file_path, mode == HISTORY_WRITE ? "wb" : "rb");
	self.mode = mode;

	assert(self.file != NULL);

	return self;
}

void history_add_token(struct history_t *history, enum history_token_t token) {
	assert(history->mode == HISTORY_WRITE);
	fwrite(&token, sizeof(uint8_t), 1, history->file);
}

void history_add_int(struct history_t *history, integer_t integer) {
	assert(history->mode == HISTORY_WRITE);
	fwrite(&integer, sizeof(integer), 1, history->file);
}

void history_add_dec(struct history_t *history, decimal_t decimal) {
	assert(history->mode == HISTORY_WRITE);
	fwrite(&decimal, sizeof(decimal), 1, history->file);
}

enum history_token_t history_read_token(struct history_t *history) {
	assert(history->mode == HISTORY_READ);

	enum history_token_t token = HISTORY_SPACE;
	fread(&token, sizeof(uint8_t), 1, history->file);

	return token;
}

integer_t history_read_int(struct history_t *history) {
	assert(history->mode == HISTORY_READ);

	integer_t value;
	fread(&value, sizeof(integer_t), 1, history->file);
	return value;
}

decimal_t history_read_dec(struct history_t *history) {
	assert(history->mode == HISTORY_READ);

	decimal_t value;
	fread(&value, sizeof(decimal_t), 1, history->file);
	return value;
}

void history_close(struct history_t *history) {
	fclose(history->file);
}

