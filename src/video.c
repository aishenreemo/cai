#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <SDL.h>

#include "matrix.h"
#include "network.h"

#define COLOR_BACKGROUND 0x0B0F10
#define COLOR_FOREGROUND 0xC5C8C9

#define COLOR_BLACK	0x131718
#define COLOR_RED	0xDF5B61
#define COLOR_GREEN	0x87C7A1
#define COLOR_YELLOW	0xDE8F78
#define COLOR_BLUE	0x6791C9
#define COLOR_PURPLE	0xBC83E3
#define COLOR_CYAN	0x70B9CC
#define COLOR_WHITE	0xC4C4C4

#define COLOR_BRIGHT_BLACK	0x4D4747
#define COLOR_BRIGHT_RED	0xFA7781
#define COLOR_BRIGHT_GREEN	0xFFF0CB
#define COLOR_BRIGHT_YELLOW	0xEFDCCC
#define COLOR_BRIGHT_BLUE	0x8EA6B9
#define COLOR_BRIGHT_PURPLE	0xFDCBC5
#define COLOR_BRIGHT_CYAN	0xC5E1FC
#define COLOR_BRIGHT_WHITE	0xFFFCEE

#define SDL_ColorHexToArgs(COLOR) \
	(Uint8) (((COLOR) >> 16) & 0xFF),  \
	(Uint8) (((COLOR) >> 8) & 0xFF),  \
	(Uint8) (((COLOR) >> 0)  & 0xFF)

#define DEFAULT_WIN_WIDTH  800
#define DEFAULT_WIN_HEIGHT 600

void listen(SDL_Event *event);
void render(
	struct SDL_Window *window,
	struct SDL_Renderer *renderer,
	struct network_t *network
);

void SDL_RenderDrawCircle(SDL_Renderer *renderer, int center_x, int center_y, int radius);

int main(void) {
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}

	struct SDL_Window *window = SDL_CreateWindow(
		"Neural Networks",
		0,
		0,
		DEFAULT_WIN_WIDTH,
		DEFAULT_WIN_HEIGHT,
		SDL_WINDOW_UTILITY
	);

	struct SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

	// struct history_t history = history_new("dist/sample.aiv", HISTORY_READ);
	// struct network_t network = network_from_history(&history);

	// enum history_token_t token = history_read_token(&network.history);

	// while (token != HISTORY_EOF) {
	// 	SDL_Event event;
	// 	listen(&event);
	// 	if (event.type == SDL_QUIT) {
	// 		break;
	// 	}

	// 	network_read_frame(&network, &token);
	// 	render(window, renderer, &network);
	// 	// network_print(&network);

	// 	token = history_read_token(&network.history);
	// 	SDL_Delay(1000 / 15);
	// }

	// network_free(&network);
	
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}

void listen(SDL_Event *event) {
	while (SDL_PollEvent(event)) {
		if (event->type == SDL_QUIT) {
			return;
		}
	}
}

void render(
	struct SDL_Window *window,
	struct SDL_Renderer *renderer,
	struct network_t *network
) {
	SDL_SetRenderDrawColor(renderer, SDL_ColorHexToArgs(COLOR_BACKGROUND), 0xFF);
	SDL_RenderClear(renderer);

	int screen_size[2];
	SDL_GetWindowSize(window, screen_size + 0, screen_size + 1);
	int screen_length = screen_size[0] > screen_size[1] ? screen_size[0] : screen_size[1];


	int neuron_h_distance = screen_size[0] / (1 + network->layer_count);
	for (int i = 0; i < network->layer_count; i++) {
		int layer_size = network->activations[NETWORK_ORIGINAL][i].cols;
		int x = neuron_h_distance * (i + 1);
		int radius = (screen_length * 0.2) / (layer_size + 2);

		int neuron_v_distance = screen_size[1] / (1 + layer_size);

		SDL_SetRenderDrawColor(renderer, SDL_ColorHexToArgs(COLOR_FOREGROUND), 0xFF);
		for (int j = 0; j < layer_size; j++) {
			int y = neuron_v_distance * (j + 1);
			SDL_RenderDrawCircle(renderer, x, y, radius);
		}

		if (i == network->layer_count - 1) continue;
		struct matrix_t *weights = network->weights[NETWORK_ORIGINAL] + i;

		int weights_count = weights->cols * weights->rows;
		int neuron_v_distance_next = screen_size[1] / (1 + weights->cols);
		int radius_next = (screen_length * 0.2) / (weights->cols + 2);
		for (int j = 0; j < weights_count; j++) {
			int col = j % weights->cols;
			int row = j / weights->cols;
			int ix = neuron_h_distance * (i + 1) + radius;
			int iy = neuron_v_distance * (row + 1);
			int ox = neuron_h_distance * (i + 2) - radius_next;
			int oy = neuron_v_distance_next * (col + 1);

			SDL_RenderDrawLine(renderer, ix, iy, ox, oy);
		}
	}

	SDL_RenderPresent(renderer);
}

void SDL_RenderDrawCircle(SDL_Renderer *renderer, int center_x, int center_y, int radius) {
	int diameter = (radius * 2);

	int x = radius - 1;
	int y = 0;

	int t_x = 1;
	int t_y = 1;

	int error = t_x - diameter;

	while (x >= y) {
		SDL_RenderDrawPoint(renderer, center_x + x, center_y - y);
		SDL_RenderDrawPoint(renderer, center_x + x, center_y + y);
		SDL_RenderDrawPoint(renderer, center_x - x, center_y - y);
		SDL_RenderDrawPoint(renderer, center_x - x, center_y + y);
		SDL_RenderDrawPoint(renderer, center_x + y, center_y - x);
		SDL_RenderDrawPoint(renderer, center_x + y, center_y + x);
		SDL_RenderDrawPoint(renderer, center_x - y, center_y - x);
		SDL_RenderDrawPoint(renderer, center_x - y, center_y + x);

		if (error <= 0) {
			y += 1;
			error += t_y;
			t_y += 2;
		}

		if (error > 0) {
			x -= 1;
			t_x += 2;
			error += t_x - diameter;
		}
	}
}
