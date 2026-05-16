import sys, tensorrt as trt

engine_path = sys.argv[1] if len(sys.argv) > 1 else "pointpillars_io_fp16.plan"
logger = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

print(f"== Engine: {engine_path} ==")
print(f"Profiles: {engine.num_optimization_profiles}")
print(f"Bindings: {engine.num_bindings}")
for i in range(engine.num_bindings):
    name   = engine.get_binding_name(i)
    is_in  = engine.binding_is_input(i)
    dtype  = engine.get_binding_dtype(i)
    shape  = engine.get_binding_shape(i)  # 若有 -1，可用 context.get_binding_shape(i)
    io     = "INPUT " if is_in else "OUTPUT"
    print(f"[{i:02d}] {io}  name={name:20s}  dtype={dtype}  shape={tuple(shape)}")

# 如果是动态 shape，可用 context 读实际 shape：
ctx = engine.create_execution_context()
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    print(f"runtime-shape[{i:02d}] {name:20s} -> {tuple(ctx.get_binding_shape(i))}")
