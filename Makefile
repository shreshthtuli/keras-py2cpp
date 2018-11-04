INC=-I./include

default: main

main:	test.cpp src/*
		g++ -std=c++1z $(INC) -o main test.cpp

clean:
		rm -r test.o make.out