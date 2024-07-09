CC := gcc

SOURCE := ./src
DIST := ./dist

FLAGS := -Wall -O0 -march=native
LIBS := -lm

OBJECTS := $(DIST)/matrix.o $(DIST)/network.o $(DIST)/history.o
TARGETS := nn_train nn_video

all: $(DIST) $(OBJECTS) $(DIST)/train.o $(DIST)/video.o $(TARGETS)

$(DIST):
	mkdir -p $@

$(DIST)/%.o: $(SOURCE)/%.c
	$(CC) -g -c $(FLAGS) $(LIBS) $< -o $@

nn_%: $(DIST)/%.o $(OBJECTS)
	$(CC) -g $(FLAGS) $(LIBS) $^ -o $@

clean:
	rm -rf *~ $(TARGETS) $(DIST)

.PHONY: clean
