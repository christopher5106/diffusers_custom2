from transformers import AutoTokenizer, PretrainedConfig
tokenizer_one = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer",
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer_2",
    use_fast=False,
)

print(tokenizer_one("this is cool"))  # {'input_ids': [49406, 589, 533, 2077, 49407], 'attention_mask': [1, 1, 1, 1, 1]}
# print(tokenizer_one.get_input_ids("this is cool"))
print(tokenizer_one._tokenize("this is cool"))  # ['this</w>', 'is</w>', 'cool</w>']

print(tokenizer_one.convert_ids_to_tokens(tokenizer_one("this is cool")["input_ids"]))  # ['<|startoftext|>', 'this</w>', 'is</w>', 'cool</w>', '<|endoftext|>']

from special_token import modify_tokenizers
from diffusers import StableDiffusionXLPipeline
model = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
)

modify_tokenizers(model)
print(model.tokenizer("this is cool"))

