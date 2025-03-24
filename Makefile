CFLAG = -O3 -g -Wall -fopenmp
CFLAGS = -O3 -Wall -Wextra -std=c99

all: winograd

winograd: libwinograd.a driver.o
	g++ driver.o libwinograd.a -std=c++11 ${CFLAG} -o winograd

libwinograd.a: winograd-rs/src/lib.rs
	( cd winograd-rs && cargo build --release )
	cp winograd-rs/target/release/libwinograd.a .

driver.o: driver.cc winograd.h

test: winograd
	flamegraph -- ./winograd ./conf/small.conf

clean:
	rm -f winograd *.o libwinograd.a
	rm -rf winograd-rs/target/
	rm -f flamegraph.svg
	rm -f perf.data perf.data.old
