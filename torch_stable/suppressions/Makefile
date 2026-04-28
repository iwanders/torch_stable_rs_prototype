SHELL=/bin/bash

CPPFLAGS=-g -O2 -I$(VIRTUAL_ENV)/../build/ -I$(VIRTUAL_ENV)/../

#println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
#println!("cargo:rustc-link-arg=-ltorch");
#// println!("cargo:rustc-link-arg=-L{}", lib_path);
#//
#println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);


LDLIBS=-L$(VIRTUAL_ENV)/../build/lib/ -Wl,-rpath,$(VIRTUAL_ENV)/../build/lib/ -Wl,--no-as-needed -ltorch -ltorch_cpu

main:
	g++ $(CPPFLAGS) -o main main.cpp $(LDLIBS)

generate_suppressions: main
	#yes | valgrind  --gen-suppressions=yes   ./main
	valgrind --gen-suppressions=all --log-file=/tmp/generated_suppressions.supp ./main
