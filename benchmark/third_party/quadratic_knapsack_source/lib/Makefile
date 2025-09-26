# Simple makefile to compile to create a projetion library

# Compilation flags
CC=gcc
FC=gfortran
OPTFLAGS=-march=native -O3 --fast-math -fPIC
CFLAGS=-ansi -std=c99 -Wall -pedantic $(OPTFLAGS)
SHAREDFLAGS=-shared
ifeq ($(CFG),debug)
	CFLAGS += -DDEBUG
endif
FFLAGS=$(OPTFLAGS)

# Standard libraries
LIBS=-lm

# Targets
LIBRARY=newtonproj.so
OBJS=cont_quad_knapsack.o third_party_methods.o newton_projection.o
all: $(LIBRARY) $(OBJS)

# Dependencies and build rules
cont_quad_knapsack.o: cont_quad_knapsack.h cont_quad_knapsack.c
	$(CC) $(CFLAGS) -c cont_quad_knapsack.c

third_party_methods.o: third_party_methods.h third_party_methods.c
	$(CC) $(CFLAGS) -c third_party_methods.c

newton_projection.o: cont_quad_knapsack.h newton_projection.c
	$(CC) $(CFLAGS) -c newton_projection.c

$(LIBRARY): $(OBJS)
	$(CC) $(CFLAGS) $(SHAREDFLAGS) -o $@ $^

clean:
	rm $(OBJS) $(LIBRARY) 
