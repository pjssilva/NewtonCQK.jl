CCP:=g++

export

hexaly_cqk.so: hexaly_cqk.cpp
	$(CCP) hexaly_cqk.cpp -shared -I $(hexaly_path)/include -lhexaly145 -lpthread -o hexaly_cqk.so

clean:
	rm *.so *~
