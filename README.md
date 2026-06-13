# FLARE — What does π0 actually do with your instruction?

Studying how π0 fine-tuned on LIBERO use, and bypass, their language instruction.

Model rollouts, attention capture, probing, steering, and LoRA require a GPU with the openpi environment (the `pi0_libero` checkpoint) plus LIBERO.

LoRA adapters were trained with openpi's standard adapter configuration on single-task LIBERO demonstrations (3,000 steps)
