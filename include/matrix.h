#ifndef MATRIX_H
#define MATRIX_H

#ifndef MATRIX_ITEM
#define MATRIX_ITEM double
typedef MATRIX_ITEM matrix_item_t;
#endif // MATRIX_ITEM

struct matrix_t {
	int cols;
	int rows;
	int stride;

	matrix_item_t *items;
};

struct matrix_t matrix_new(int cols, int rows);
struct matrix_t matrix_from(matrix_item_t *items, int cols, int rows, int stride);

void matrix_sum(struct matrix_t *dst, struct matrix_t *const src);
void matrix_mul(struct matrix_t *dst, struct matrix_t *const a, struct matrix_t *const b);
void matrix_fill(struct matrix_t *dst, matrix_item_t value);

void matrix_rand(struct matrix_t *matrix);

void matrix_print(struct matrix_t *const matrix);

void matrix_free(struct matrix_t *matrix);

#define MATRIX_AT(M, COL, ROW) (M).items[(ROW) * (M).stride + (COL)]

#endif // MATRIX_H
