#ifndef HISTORY_H
#define HISTORY_H

#include <stdint.h>
#include <stdio.h>

#include "matrix.h"

enum history_mode_t {
	HISTORY_WRITE,
	HISTORY_READ,
};

enum history_token_t {
	HISTORY_SPACE,
	HISTORY_COMMA,
	HISTORY_END,
	HISTORY_LC,
	HISTORY_NC,
	HISTORY_A,
	HISTORY_W,
	HISTORY_B,
	HISTORY_N,
	HISTORY_EOF,
};

struct history_t {
	enum history_mode_t mode;
	FILE *file;
};

struct history_t history_new(char const *file_path, enum history_mode_t);
void history_add_token(struct history_t *history, enum history_token_t token);
void history_add_int(struct history_t *history, integer_t integer);
void history_add_dec(struct history_t *history, decimal_t decimal);

enum history_token_t history_read_token(struct history_t *history);
integer_t history_read_int(struct history_t *history);
decimal_t history_read_dec(struct history_t *history);

void history_close(struct history_t *history);

#endif // !HISTORY_H
