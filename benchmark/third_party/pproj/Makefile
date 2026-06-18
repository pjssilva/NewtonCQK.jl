default: ccode

PPROJ = PPROJ-1.0
SUITESPARSEDIR = $(PPROJ)/SuiteSparse-4.4.7
BLAS = -lopenblas -lgfortran
LAPACK =

include $(SUITESPARSEDIR)/SuiteSparse_config/SuiteSparse_config.mk

# C code only:
ccode: pproj_cqk

I = -I$(SUITESPARSEDIR)/CCOLAMD/Include \
	-I$(SUITESPARSEDIR)/CHOLMOD/Include \
	-I$(SUITESPARSEDIR)/metis-4.0/Lib \
	-I$(SUITESPARSEDIR)/AMD/Include \
	-I$(SUITESPARSEDIR)/COLAMD \
	-I$(SUITESPARSEDIR)/CAMD/Include \
	-I$(SUITESPARSEDIR)/SuiteSparse_config -I..

# with no debugging
CFLAGS = -O3 -std=c99 -lpthread
CC = gcc
C = $(CC) $(CFLAGS) -DNDEBUG $(I)

LIB = $(PPROJ)/pproj.a \
	$(PPROJ)/Check/check.a \
	$(SUITESPARSEDIR)/CHOLMOD/Lib/libcholmod.a \
	$(SUITESPARSEDIR)/AMD/Lib/libamd.a \
	$(SUITESPARSEDIR)/CAMD/Lib/libcamd.a \
	$(SUITESPARSEDIR)/CCOLAMD/Lib/libccolamd.a \
	$(SUITESPARSEDIR)/COLAMD/Lib/libcolamd.a \
	$(SUITESPARSEDIR)/metis-4.0/libmetis.a \
	$(SUITESPARSEDIR)/SuiteSparse_config/libsuitesparseconfig.a \
	$(LAPACK) $(BLAS) $(XERBLA) -lm

pproj_cqk: pproj_cqk.o
	$(C) -shared -o pproj_cqk pproj_cqk.o $(LIB)

pproj_cqk.o:
	$(C) -c $*.c

clean:
	rm *.o pproj_cqk
