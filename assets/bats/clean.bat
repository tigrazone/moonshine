@echo off
pushd %CD%
cd ..\..
del .zig-cache\*.* /Q /S
rd  /Q /S .zig-cache
del zig-out\*.* /Q /S
rd  /Q /S zig-out
popd