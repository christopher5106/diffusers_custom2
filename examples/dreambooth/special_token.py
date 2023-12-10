import torch
from torch import nn
from typing import Optional


class CLIPTextEmbeddingsSpecialToken(nn.Module):
    def __init__(self, clip_text_embeddings):
        super().__init__()
        self.subnet = clip_text_embeddings
        embed_dim = self.subnet.token_embedding.embedding_dim  # 768, 1280 (for each encoder)
        self.special_token_embedding = torch.nn.Parameter(
            torch.zeros((1, 1, embed_dim), dtype=torch.float32)
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        assert position_ids is None
        assert inputs_embeds is None  # TODO has been not sliced because it's mainly None
        embeddings = torch.cat([
            self.special_token_embedding.repeat(1, 1, 1),  # TODO repeat with correct batch size instead of 1
            self.subnet(input_ids[:, 1:], position_ids, inputs_embeds)  # (batch size, position, emb_size)
        ], dim=1)
        return embeddings


def add_special_token(text_encoder):
    special_embeddings = CLIPTextEmbeddingsSpecialToken(text_encoder.text_model.embeddings)
    text_encoder.text_model.embeddings = special_embeddings
    return [special_embeddings.special_token_embedding]


def load_special_token(model):
    state_dict, network_alphas = model.lora_state_dict(lora_path)
    assert network_alphas is None
    with torch.no_grad():
        text_specialtoken_parameters_one = add_special_token(model.text_encoder)
        text_specialtoken_parameters_two = add_special_token(model.text_encoder_2)
        text_specialtoken_parameters_one[0].copy_(state_dict["text_encoder.special_token_embedding"])
        text_specialtoken_parameters_two[0].copy_(state_dict["text_encoder_2.special_token_embedding"])
    del state_dict["text_encoder.special_token_embedding"]
    del state_dict["text_encoder_2.special_token_embedding"]
    return state_dict
