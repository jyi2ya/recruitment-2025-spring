CFLAG = -O3 -g -Wall -fopenmp
CFLAGS = -O3 -Wall -Wextra -std=c99
CXX ?= g++
RUST_TARGET ?= x86_64-unknown-linux-musl

all: winograd

winograd: libwinograd.a driver.o
	$(CXX) -static driver.o libwinograd.a -std=c++11 ${CFLAG} -o winograd

winograd-debug: libwinograd-debug.a driver.o
	$(CXX) driver.o libwinograd-debug.a -std=c++11 ${CFLAG} -o winograd-debug

libwinograd.a: winograd-rs/src/lib.rs
	( cd winograd-rs && cargo build --target $(RUST_TARGET) --release )
	cp winograd-rs/target/$(RUST_TARGET)/release/libwinograd.a .

libwinograd-debug.a: winograd-rs/src/lib.rs
	( cd winograd-rs && cargo build --target $(RUST_TARGET) )
	cp winograd-rs/target/$(RUST_TARGET)/debug/libwinograd.a ./libwinograd-debug.a

driver.o: driver.cc winograd.h

debug-test: winograd-debug
	./winograd-debug ./conf/small.conf 1

debug-bench: winograd-debug
	./winograd-debug ./conf/small.conf 0

test: winograd
	./winograd ./conf/small.conf 1

bench: winograd
	./winograd ./conf/small.conf 0

bigtest: winograd
	./winograd ./conf/vgg16.conf 0

profile: winograd
	flamegraph -- ./winograd ./conf/small.conf

clean:
	rm -f winograd winograd-debug *.o libwinograd.a libwinograd-debug.a
	rm -rf winograd-rs/target/
	rm -f flamegraph.svg
	rm -f perf.data perf.data.old
