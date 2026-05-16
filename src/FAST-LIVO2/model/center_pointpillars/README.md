# CenterPoint Model Assets

This directory intentionally does not contain generated model binaries.

To enable the optional CenterPoint node, place the required TensorRT/ONNX
artifacts here or point the launch/config files to your own model directory.
Typical required files are:

- `rpn_centerhead_sim.plan`
- `centerpoint.scn.onnx`

Large generated artifacts should stay out of git.
