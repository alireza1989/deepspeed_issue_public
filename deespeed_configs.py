STAGE1 = {
    "zero_optimization":
    {
        "stage": 1,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "allgather_bucket_size": 200000000,
        "reduce_bucket_size": 200000000,
        "sub_group_size": 1000000000000
    },
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "synchronize_checkpoint_boundary": false
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "single_submit": false,
        "overlap_events": true,
        "thread_count": 1
    },
    "gradient_accumulation_steps": 128,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 1
}

STAGE2 = {
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },

    "steps_per_print": 2000,
    "wall_clock_breakdown": False,
    "train_micro_batch_size_per_gpu": 4
}

STAGE3 = {
    "flops_profiler": {
        "enabled": True,
        "profile_step": 1,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 0.94e6,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": False,
    "train_micro_batch_size_per_gpu": 1
}