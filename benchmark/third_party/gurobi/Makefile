CC:=gcc

export

gurobi_cqk.so: gurobi_cqk.c
	$(CC) -g -m64 -shared -o gurobi_cqk.so gurobi_cqk.c -I $(gurobi_path)/include/ -L $(gurobi_path)/lib -lgurobi130 -lm

clean:
	rm *.so
