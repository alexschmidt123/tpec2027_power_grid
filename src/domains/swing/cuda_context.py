"""Share the device primary CUDA context with PyTorch."""
import pycuda.driver as cuda
_primary_ctx=None

def ensure_primary_context():
    global _primary_ctx
    cuda.init()
    if _primary_ctx is None:
        import torch
        dev_id=int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        _primary_ctx=cuda.Device(dev_id).retain_primary_context()
    return _primary_ctx

class _PrimaryCtx:
    def __enter__(self):
        ensure_primary_context().push()
        return _primary_ctx
    def __exit__(self,*exc):
        _primary_ctx.pop()
        return False
