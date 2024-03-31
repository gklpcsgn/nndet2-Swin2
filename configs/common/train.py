# if you dont know the epoch logic 
# https://github.com/facebookresearch/detectron2/discussions/3622
train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=100,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=10,
    log_period=20,
    device="cuda"
    # ...
)
