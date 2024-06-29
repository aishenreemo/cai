#ifndef MATRIX_H
#define MATRIX_H

#ifndef MATRIX_DECIMAL
#define MATRIX_DECIMAL double
typedef MATRIX_DECIMAL decimal_t;
#endif

struct matrix_t {
	int cols;
	int rows;
	int stride;
	decimal_t *items;
};

struct matrix_t matrix_new(int cols, int rows);
struct matrix_t matrix_from(decimal_t *items, int cols, int rows, int stride);

void matrix_rand(struct matrix_t *matrix);
void matrix_fill(struct matrix_t *matrix, decimal_t value);

void matrix_add(struct matrix_t *sum, struct matrix_t *const a, struct matrix_t *const b);
void matrix_mul(struct matrix_t *product, struct matrix_t *const a, struct matrix_t *const b);

void matrix_print(struct matrix_t *matrix);

#define MATRIX_AT(M, COL, ROW) (M).items[(ROW) * (M).stride + (COL)]

#endif // !MATRIX_H
