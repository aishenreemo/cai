#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "matrix.h"

struct matrix_t matrix_new(int cols, int rows) {
	struct matrix_t self;
	self.cols = cols;
	self.rows = rows;
	self.stride = cols;

	self.items = calloc(cols * rows, sizeof(decimal_t));
	return self;
}

struct matrix_t matrix_from(decimal_t *items, int cols, int rows, int stride) {
	struct matrix_t matrix;

	matrix.cols = cols;
	matrix.rows = rows;
	matrix.stride = stride;
	matrix.items = items;

	return matrix;
}

void matrix_add(
	struct matrix_t *sum,
	struct matrix_t *const a,
	struct matrix_t *const b
) {
	assert(sum->cols == a->cols);
	assert(sum->rows == a->rows);
	assert(sum->cols == b->cols);
	assert(sum->rows == b->rows);

	for (int i = 0; i < sum->rows; i++) {
		for (int j = 0; j < sum->cols; j++) {
			MATRIX_AT(*sum, j, i) = MATRIX_AT(*a, j, i) + MATRIX_AT(*b, j, i);
		}
	}
}

void matrix_mul(
	struct matrix_t *product,
	struct matrix_t *const a,
	struct matrix_t *const b
) {
	assert(a->cols == b->rows);
	assert(product->rows == a->rows);
	assert(product->cols == b->cols);

	int length = product->rows * product->cols;
	for (size_t i = 0; i < length; i++) {
		int col = i % product->cols;
		int row = i / product->cols;

		MATRIX_AT(*product, col, row) = 0;
		for (size_t k = 0; k < a->cols; k++) {
			MATRIX_AT(*product, col, row) +=
				MATRIX_AT(*a, k, row) *
				MATRIX_AT(*b, col, k);
		}
	}
}

void matrix_fill(struct matrix_t *dst, decimal_t value) {
	for (int i = 0; i < dst->rows; i++) {
		for (int j = 0; j < dst->cols; j++) {
			MATRIX_AT(*dst, j, i) = value;
		}
	}
}

void matrix_print(struct matrix_t *matrix) {
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->cols; j++) {
			printf("%lf ", MATRIX_AT(*matrix, j, i));
		}
		printf("\n");
	}
}

void matrix_rand(struct matrix_t *dst) {
	for (int i = 0; i < dst->rows; i++) {
		for (int j = 0; j < dst->cols; j++) {
			MATRIX_AT(*dst, j, i) = (decimal_t) rand() / (decimal_t) RAND_MAX;
		}
	}
}
