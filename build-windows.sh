#!/bin/sh
set -eu
IFS=$(printf "\n\t")
# scratch=$(mktemp -d -t tmp.XXXXXXXXXX)
# atexit() {
#   rm -rf "$scratch"
# }
# trap atexit EXIT

( cd winograd-rs && cargo build --target x86_64-pc-windows-gnu --release )
x86_64-w64-mingw32-g++ -O3 -static driver.cc winograd-rs/target/x86_64-pc-windows-gnu/release/libwinograd.a  ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/winapi-x86_64-pc-windows-gnu-0.4.0/lib/*.a -o winograd.exe && mv winograd.exe /mnt/d/winograd/
