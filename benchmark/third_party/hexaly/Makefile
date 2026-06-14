CCP:=g++

export

hexaly_cqk.so: hexaly_cqk.cpp
	$(CCP) -m64 -fPIC -shared hexaly_cqk.cpp -o hexaly_cqk.so -I $(hexaly_path)/include -lhexaly145 -lpthread

clean:
	rm *.so
