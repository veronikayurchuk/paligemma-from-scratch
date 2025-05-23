from typing import Optional, Tuple
import torch
import torch.nn as nn
from configs import SiglipVisionConfig


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size, # so no overlap between patches
            padding=0,
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embedding_dim)
        self.register_buffer("position_ids", 
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, H, W = pixel_values.shape # [batch_size, channels, H, W]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since 
        # the stride is equal to the kernel size
        # The output of the convolution will have shape [batch_size, embed_dim, num_patches_h, num_patches_w]
        # where num_patches_h = H // patch_size and num_patches_w = W // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [batch_size, embed_dim, num_patches_h, num_patches_w] -> [batch_size, embed_dim, num_patches]
        # where num_patches = num_patches_h * num_patches_w
        embeddings = patch_embeds.flatten(2)
        # [batch_size, embed_dim, num_patches] -> [batch_size, num_patches, embed_dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [batch_size, num_patches, embed_dim]
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [batch_size, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden_states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [batch_size, num_patches, embed_dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [batch_size, num_patches, embed_dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [batch_size, num_patches, embed_dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [batch_size, num_patches, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate the attention using the formula Q * K^T / sqrt(d_k). 
        #  Q * K^T: [batch_size, num_patches, num_heads, head_dim] * [batch_size, num_patches, head_dim, num_heads] -> [batch_size, num_patches, num_heads, num_patches]
        # attn_weights: [batch_size, num_patches, num_heads, num_patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [batch_size, num_patches, num_heads, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. 
        # attn_weights * value_states: [batch_size, num_patches, num_heads, num_patches] * [batch_size, num_patches, num_heads, head_dim] 
        # -> [batch_size, num_patches, num_heads, head_dim]
        # attn_output: [batch_size, num_patches, num_heads, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)


        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [batch_size, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipEncoderLayer(nn.Module):
    """
    SiglipEncoderLayer is a single encoder layer for vision tasks.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states
        # residual: [batch_size, num_patches, embed_dim] 
        residual = hidden_states
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        # [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipVisionEncoder(nn.Module):
    """
    SiglipVisionEncoder is a transformer encoder for vision tasks.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [batch_size, num_patches, embed_dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states
        


class SiglipVisionTransformer(nn.Module):
    """
    SiglipVisionTransformer is a transformer encoder for vision tasks.
    """
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values: torch.Tensor):
        # [batch_size, num_channels, H, W] -> [batch_size, num_patches, embed_dim]
        return self.vision_model(pixel_values)


