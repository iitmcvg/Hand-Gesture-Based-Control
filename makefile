# the compiler
CC = g++

# the compiler flags
CFLAGS = `pkg-config opencv --cflags`
LIBS =`pkg-config opencv --libs` -I.

all: handtrack

handtrack: handtrack.cpp instructions.cpp
	$(CC) $(CFLAGS) -o output handtrack.cpp instructions.cpp $(LIBS)





