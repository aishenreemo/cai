CC := gcc

SOURCE := ./src
DIST := ./dist
INCLUDE := ./include

FLAGS := -Wall -O0 -march=native -I$(INCLUDE)
LIBS := -lm

OBJECTS := $(DIST)/main.o $(DIST)/matrix.o $(DIST)/network.o
TARGET := cai

all: $(DIST) $(OBJECTS) $(TARGET)

$(DIST):
	mkdir -p $@

$(DIST)/%.o: $(SOURCE)/%.c
	$(CC) -g -c -o $@ $< $(FLAGS) $(LIBS)

$(TARGET): $(OBJECTS)
	$(CC) -g -o $@ $^ $(FLAGS) $(LIBS)

clean:
	rm -rf *~ $(TARGET) $(DIST)

.PHONY: clean
