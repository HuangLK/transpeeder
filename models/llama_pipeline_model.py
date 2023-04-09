
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
import deepspeed
from deepspeed.pipe import PipelineModule


def _wrap_embed_layer(layer: torch.nn.Module):
    class EmbeddingPipe(torch.nn.Embedding):
        def forward(self, args):
            input_ids, attention_mask, _ = args
            inputs_embeds = super().forward(input_ids)
            return (inputs_embeds, attention_mask)

    layer.__class__ = EmbeddingPipe
    return layer

def _wrap_decoder_layer(layer: torch.nn.Module, activation_checkpointing=False):
    class ParallelTransformerLayerPipe(LlamaDecoderLayer):
        def forward(self, args):
            hidden_states, mask = args
            attention_mask = torch.where(mask == True, float("-inf"), 0).long()
            # TODO: past_key_value, use_cache
            outputs = LlamaDecoderLayer.forward(self, hidden_states, attention_mask)
            return (outputs[0], mask)

        def _ckpt_forward(self, args):
            hidden_states, mask = args
            attention_mask = torch.where(mask == True, float("-inf"), 0).long()

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return LlamaDecoderLayer.forward(module, *inputs)
                return custom_forward

            # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
            outputs = deepspeed.checkpointing.checkpoint(
                create_custom_forward(self),
                hidden_states,
                attention_mask,
            )

            return (outputs, mask)

    if activation_checkpointing:
        ParallelTransformerLayerPipe.forward = ParallelTransformerLayerPipe._ckpt_forward

    layer.__class__ = ParallelTransformerLayerPipe
    return layer

def _wrap_norm_layer(layer: torch.nn.Module):
    class LayerNormPipe(LlamaRMSNorm):
        def forward(self, args):
            hidden_states, _ = args
            last_hidden_states = super().forward(hidden_states)
            return (last_hidden_states,)

    layer.__class__ = LayerNormPipe
    return layer

def _wrap_lm_layer(layer: torch.nn.Module):
    class LMLayerPipe(torch.nn.Linear):
        def forward(self, args):
            hidden_states, = args
            logits = super().forward(hidden_states)
            return (logits,)

    layer.__class__ = LMLayerPipe
    return layer

def _to_layers(lm_model, activation_checkpointing=False):
    layers = [
        _wrap_embed_layer(lm_model.model.embed_tokens),
        *[_wrap_decoder_layer(layer, activation_checkpointing) for layer in lm_model.model.layers],
        _wrap_norm_layer(lm_model.model.norm),
        _wrap_lm_layer(lm_model.lm_head),
    ]
    return layers


def loss_fn(outputs, labels):
    # unpack
    logits, = outputs
    # all labels are `ignore_index` will cause nan
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
    )


def get_model(lm_model, args, activation_checkpointing_config=None):
    class GPT2ModelPipe(PipelineModule):
        def __init__(self, lm_model, **kwargs):
            if activation_checkpointing_config:
                deepspeed.checkpointing.configure(
                    None,
                    partition_activations=activation_checkpointing_config.get("partition_activations", False),
                    contiguous_checkpointing=activation_checkpointing_config.get("contiguous_memory_optimization", False),
                    checkpoint_in_cpu=activation_checkpointing_config.get("cpu_checkpointing", False),
                    num_checkpoints=activation_checkpointing_config.get("number_checkpoints", None),
                    synchronize=activation_checkpointing_config.get("synchronize_checkpoint_boundary", False),
                    profile=activation_checkpointing_config.get("profile", False),
                )
            super().__init__(
                layers=_to_layers(lm_model, activation_checkpointing=activation_checkpointing_config is not None),
                **kwargs
            )

    pp = args.pipe_parallel_size
    mp = args.model_parallel_size
    assert args.world_size % (pp * mp) == 0
    dp = args.world_size // (pp * mp)

    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)
    # Offset base seeds for the interior pipeline stages.
    stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
    if 0 < stage_id < topo.get_dim('pipe') - 1:
        args.seed = args.seed + (stage_id * mp)

    return GPT2ModelPipe(lm_model,
                         loss_fn=loss_fn,
                         topology=topo,
                         base_seed=args.seed,)
