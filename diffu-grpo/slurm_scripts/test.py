from transformers import AutoTokenizer, AutoModel
import torch    

model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Base',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
print(model)