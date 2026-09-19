import importlib.util, traceback

spec = importlib.util.spec_from_file_location('mod', '../Untitled-1.py'.replace('../', ''))
# Use workspace-relative path
spec = importlib.util.spec_from_file_location('mod', 'Untitled-1.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    df = mod.scan_top_derivative_assets()
    print('DF_SHAPE:', getattr(df, 'shape', None))
    print(df.head().to_string())
except Exception:
    traceback.print_exc()
