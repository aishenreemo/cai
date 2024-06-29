CC := gcc

SOURCE := ./src
DIST := ./dist

FLAGS := -Wall -O0 -march=native
LIBS := -lm

OBJECTS := $(DIST)/main.o $(DIST)/matrix.o $(DIST)/network.o
TARGET := cai

all: $(DIST) $(OBJECTS) $(TARGET)

$(DIST):
	mkdir -p $@

$(DIST)/%.o: $(SOURCE)/%.c
	$(CC) -g -c $(FLAGS) $(LIBS) $< -o $@

$(TARGET): $(OBJECTS)
	$(CC) -g $(FLAGS) $(LIBS) $^ -o $@

clean:
	rm -rf *~ $(TARGET) $(DIST)

.PHONY: clean
