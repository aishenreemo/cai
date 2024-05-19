#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "matrix.h"


struct matrix_t matrix_new(int cols, int rows) {
	struct matrix_t matrix;

	matrix.cols = cols;
	matrix.rows = rows;
	matrix.stride = cols;

	matrix.items = calloc(cols * rows, sizeof(matrix_item_t));

	return matrix;
}

struct matrix_t matrix_from(matrix_item_t *items, int cols, int rows, int stride) {
	struct matrix_t matrix;

	matrix.cols = cols;
	matrix.rows = rows;
	matrix.stride = stride;
	matrix.items = items;

	return matrix;
}

void matrix_sum(struct matrix_t *dst, struct matrix_t *const src) {
	assert(dst->cols == src->cols);
	assert(dst->rows == src->rows);

	for (int i = 0; i < dst->rows; i++) {
		for (int j = 0; j < dst->cols; j++) {
			MATRIX_AT(*dst, j, i) += MATRIX_AT(*src, j, i);
		}
	}
}

void matrix_mul(struct matrix_t *dst, struct matrix_t *const a, struct matrix_t *const b) {
	assert(a->cols == b->rows);
	assert(dst->rows == a->rows);
	assert(dst->cols == b->cols);

	int length = dst->rows * dst->cols;
	for (size_t i = 0; i < length; i++) {
		int col = i % dst->cols;
		int row = i / dst->cols;

		MATRIX_AT(*dst, col, row) = 0;
		for (size_t k = 0; k < a->cols; k++) {
			MATRIX_AT(*dst, col, row) += MATRIX_AT(*a, k, row) * MATRIX_AT(*b, col, k);
		}
	}
}

void matrix_fill(struct matrix_t *dst, matrix_item_t value) {
	for (int i = 0; i < dst->rows; i++) {
		for (int j = 0; j < dst->cols; j++) {
			MATRIX_AT(*dst, j, i) = value;
		}
	}
}

void matrix_rand(struct matrix_t *dst) {
	for (int i = 0; i < dst->rows; i++) {
		for (int j = 0; j < dst->cols; j++) {
			MATRIX_AT(*dst, j, i) = (matrix_item_t) rand() / (matrix_item_t) RAND_MAX;
		}
	}
}

void matrix_print(struct matrix_t *matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->cols; j++) {
			printf("\t%lf", MATRIX_AT(*matrix, j, i));
		}
		printf("\n");
	}
}

void matrix_free(struct matrix_t *matrix) {
	matrix_print(matrix);
	free(matrix->items);
}
