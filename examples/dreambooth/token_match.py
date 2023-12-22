import torch
model = "/mnt/c/Users/chris/Downloads/special_token_training_e500/pytorch_lora_weights.safetensors"
# torch.load(model)

# exit()
from safetensors import safe_open
with safe_open(model, framework="pt", device="cpu") as f:
   # for key in f.keys():
   #     print(key)
       # tensors[key] = f.get_tensor(key)
   special_token_embedding = f.get_tensor("text_encoder.special_token_embedding") # text_encoder_2.special_token_embedding

special_token_embedding = special_token_embedding.reshape((-1,))

from transformers import AutoTokenizer, PretrainedConfig
tokenizer_one = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer",
    use_fast=False,
)
# tokenizer_two = AutoTokenizer.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     subfolder="tokenizer_2",
#     use_fast=False,
# )
from transformers import CLIPTextModel
text_encoder_one = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder"
    )

print(special_token_embedding.shape)

token_embeddings = text_encoder_one.text_model.embeddings.token_embedding.weight
print(token_embeddings.shape)

token_embeddings_norm = torch.linalg.vector_norm(token_embeddings, dim=1, keepdim=True)
res = torch.matmul(token_embeddings/token_embeddings_norm, special_token_embedding).detach().numpy()
print(res.shape)

import numpy as np
token_ids = np.argsort(-res)

for token in token_ids[:200]:

    t = tokenizer_one.convert_ids_to_tokens([token])[0]
    if True or t.endswith("</w>"):
        print(t)


