# spconv Runtime

Prebuilt `libspconv.so` binaries are intentionally not committed.

When building the optional CenterPoint node, provide spconv with:

```bash
catkin_make --cmake-args -DBUILD_CENTERPOINT=ON -DSPCONV_ROOT=/path/to/libspconv
```

`SPCONV_ROOT` should contain the expected `include/` and `lib/<arch>/`
layout used by `CMakeLists.txt`.
