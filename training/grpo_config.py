MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

GRPO_CONFIG = {
    "num_generations": 4,
    "max_new_tokens": 256,
    "temperature": 0.8,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "gradient_accumulation_steps": 4,
}
