from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch    

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-1.5',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
print(model)