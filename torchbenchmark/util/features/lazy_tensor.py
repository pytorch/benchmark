import lazy_tensor_core

def has_lazy_tensor() -> bool:
    try:
        import lazy_tensor_core
        lazy_tensor_core._LAZYC._ltc_init_ts_backend()
        from caffe2.python import workspace
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=-4'])
        return True
    except ImportError:
        print("Lazy tensor core is not detected. Can't run workload with lazy tensor.")
        return False

def lazy_tensor_step_ondemand(device):
    if device == 'lazy':
        lazy_tensor_core.core.lazy_model.mark_step()