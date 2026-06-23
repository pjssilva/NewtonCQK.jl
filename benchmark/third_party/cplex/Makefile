SYSTEM := x86-64_linux
LIBFORMAT := static_pic

CC := gcc

CLNDIRS := -L $(cplex_path)/lib/$(SYSTEM)/$(LIBFORMAT)
CLNFLAGS := -lcplex -lm -lpthread -ldl

CFLAGS := -m64 -fPIC -I $(cplex_path)/include

export

cplex_cqk.so: cplex_cqk.c
	$(CC) $(CFLAGS) -shared $(CLNDIRS) -o cplex_cqk.so cplex_cqk.c $(CLNFLAGS)

clean :
	rm *.so
