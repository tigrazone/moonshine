# Chess with ray tracing - please think of a better name

### Build dependencies:
* Zig
* dxc on your path

### Possible optimizations
* Better memory/buffers
* Create homebrew version of `std.MultiArrayList` that has len as a `u32`, as that's what a `DeviceSize` is

### Random thoughts
* Orthographic projection might look visually interesting in this context 

### // TODO
* Make sure we have all necessary `errdefer`s
* Proper asset system - load scene from file rather than hardcoded
* Differentiate game and render logic better
* Swap off of GLFW or use better Zig GLFW wrapper
* **Add UI**
  * Q: Render inside UI or have UI as seperate console?
  * What does UI set?
    * Set scene
    * Set background
    * Set sample rate
    * Set camera settings (ortho vs persp)
  * What does UI Display?
    * Perf stuff
    * Current camera settings
    * Current scene info
## Some light reading
- [Importance sampling](https://computergraphics.stackexchange.com/q/4979)
- [Explicit light sampling](https://computergraphics.stackexchange.com/q/5152)
- [Multiple importance sampling](https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf)
- [Microfacets](https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/)
- [Actual materials](https://github.com/wdas/brdf) - ton of BRDF examples, in **CODE**!
- [Better sky](https://sebh.github.io/publications/egsr2020.pdf)
