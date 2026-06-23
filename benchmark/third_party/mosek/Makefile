CC=gcc
IPATHS=-I$(mosek_path)/tools/platform/linux64x86/h
LPATHS=-L$(mosek_path)/tools/platform/linux64x86/bin -Wl,-rpath-link,$(mosek_path)/tools/platform/linux64x86/bin '-Wl,-rpath=$$ORIGIN/$(mosek_path)/tools/platform/linux64x86/bin'
LIBS=-lm

mosek_cqk: mosek_cqk.c
	$(CC) -m64 -shared -fPIC $(IPATHS) $(LPATHS) -o mosek_cqk.so mosek_cqk.c -lmosek64 $(LIBS)

.PHONY: clean all

all: mosek_cqk

clean:
	rm -f *.so
