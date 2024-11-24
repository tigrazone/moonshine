@echo off
pushd %CD%
cd ..\..
zig build install-online -Doptimize=Debug
popd