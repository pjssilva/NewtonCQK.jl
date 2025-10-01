# Generate dynamic library to call Condat code from Julia

condat_simplexproj.so: condat_simplexproj.c
	gcc -march=native -O2 -fPIC -shared condat_simplexproj.c -o condat_simplexproj.so -lm

clean:
	rm *.so *~ 
