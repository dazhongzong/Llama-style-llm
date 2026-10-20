from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

model_dir = Path("models/SmolLM2-135M-Instruct")
output_dir = Path("tests/reference/output")
output_dir.mkdir(parents=True, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    torch_dtype=torch.float32,
    attn_implementation="eager",
    local_files_only=True,
)

model.eval()

tokens = [1, 100, 200]

with torch.inference_mode():
    for pos in range(len(tokens)):
        input_ids = torch.tensor(
            [tokens[:pos + 1]],
            dtype=torch.long
        )

        output = model(
            input_ids=input_ids,
            use_cache=False
        )

        logits = (
            output.logits[0, -1]
            .float()
            .cpu()
            .numpy()
        )

        file_path = output_dir / f"logits_{pos}.bin"

        logits.tofile(file_path)

        print(f"Position = {pos}")
        print(f"Logits Shape = {logits.shape}")
        print(f"Top1 Token = {np.argmax(logits)}")
        print(f"First 5 Logits = {logits[:5]}")
        print()