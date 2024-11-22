@echo off
pushd %CD%
cd ..\..
zig build install-online -Doptimize=ReleaseSmall
popd