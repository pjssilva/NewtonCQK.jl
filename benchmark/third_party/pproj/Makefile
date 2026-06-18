default: ccode

PPROJDIR = PPROJ-1.0
SUITESPARSEDIR = $(PPROJDIR)/SuiteSparse-4.4.7
BLAS = -lopenblas -lgfortran
LAPACK =

MKLROOT=

export

include $(SUITESPARSEDIR)/SuiteSparse_config/SuiteSparse_config.mk

# C code only:
ccode: pproj.a pproj_cqk

I = -I$(SUITESPARSEDIR)/CCOLAMD/Include \
	-I$(SUITESPARSEDIR)/CHOLMOD/Include \
	-I$(SUITESPARSEDIR)/metis-4.0/Lib \
	-I$(SUITESPARSEDIR)/AMD/Include \
	-I$(SUITESPARSEDIR)/COLAMD \
	-I$(SUITESPARSEDIR)/CAMD/Include \
	-I$(SUITESPARSEDIR)/SuiteSparse_config -I..

# with no debugging
CFLAGS = -O3 -std=c99 -lpthread -fPIC
CC = gcc
C = $(CC) $(CFLAGS) -DNDEBUG $(I)

LIB = pproj.a \
	$(PPROJDIR)/Check/check.a \
	pproj.a \
	$(SUITESPARSEDIR)/CHOLMOD/Lib/libcholmod.a \
	$(SUITESPARSEDIR)/AMD/Lib/libamd.a \
	$(SUITESPARSEDIR)/CAMD/Lib/libcamd.a \
	$(SUITESPARSEDIR)/CCOLAMD/Lib/libccolamd.a \
	$(SUITESPARSEDIR)/COLAMD/Lib/libcolamd.a \
	$(SUITESPARSEDIR)/metis-4.0/libmetis.a \
	$(SUITESPARSEDIR)/SuiteSparse_config/libsuitesparseconfig.a \
	$(LAPACK) $(BLAS) $(XERBLA) -lm

INCLUDE = Makefile $(PPROJDIR)/pproj.h \
	$(PPROJDIR)/Check/check.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_blas.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_check.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_cholesky.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_complexity.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_config.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_core.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_internal.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_matrixops.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_modify.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_partition.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_supernodal.h \
	$(SUITESPARSEDIR)/CHOLMOD/Include/cholmod_template.h \
	$(SUITESPARSEDIR)/AMD/Include/amd.h \
	$(SUITESPARSEDIR)/AMD/Include/amd_internal.h \
	$(SUITESPARSEDIR)/CCOLAMD/Include/ccolamd.h \
	$(SUITESPARSEDIR)/SuiteSparse_config/SuiteSparse_config.h \
	$(SUITESPARSEDIR)/COLAMD/Include/colamd.h

OBJ = pproj.o \
	pproj_initlevels.o \
	pproj_initwork.o \
	pproj_phase1.o \
	pproj_coor_ascent.o \
	pproj_dasa.o \
	pproj_sparsa.o \
	pproj_ssor0.o \
	pproj_ssor1.o

$(OBJ): $(INCLUDE)
	$(C) -c $(PPROJDIR)/$*.c

pproj_cqk: config lib $(INCLUDE) pproj_cqk.o
	$(C) -shared -o pproj_cqk.so pproj_cqk.o $(LIB)

pproj.a: $(OBJ)
	ar cr pproj.a $(OBJ)
	ranlib pproj.a

clean:
	rm *.o *.so *.a $(PPROJDIR)/Check/*.o $(PPROJDIR)/Check/check.a $(SUITESPARSEDIR)/SuiteSparse_config/*.a $(SUITESPARSEDIR)/SuiteSparse_config/libsuitesparseconfig.so.4.4.7

config:
	( cd $(SUITESPARSEDIR)/SuiteSparse_config ; $(MAKE) MKLROOT= )

lib:
	( cd $(SUITESPARSEDIR)/SuiteSparse_config/xerbla ; $(MAKE) MKLROOT= )
	( cd $(PPROJDIR)/Check ; $(MAKE) SUITESPARSEDIR=$(SUITESPARSEDIR) MKLROOT= )
	( cd $(SUITESPARSEDIR)/CHOLMOD/Lib ; $(MAKE) MKLROOT= )
	( cd $(SUITESPARSEDIR)/metis-4.0/Lib ; $(MAKE) MKLROOT= COPTIONS=-fPIC)
	( cd $(SUITESPARSEDIR)/CCOLAMD ; $(MAKE) library MKLROOT= )
	( cd $(SUITESPARSEDIR)/COLAMD ; $(MAKE) library MKLROOT= )
	( cd $(SUITESPARSEDIR)/AMD ; $(MAKE) library MKLROOT= )
	( cd $(SUITESPARSEDIR)/CAMD ; $(MAKE) library MKLROOT= )
