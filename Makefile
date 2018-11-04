main:	test.cc keras_model.cc
		g++ -std=c++11 -Wall -O3 test.cc keras_model.cc

clean:
		rm -r test.o make.out