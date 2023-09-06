
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaConfig
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec


class EmbeddingPipe(torch.nn.Embedding):
    def forward(self, args):
        input_ids, position_ids, attention_mask = args
        inputs_embeds = super().forward(input_ids)
        return (inputs_embeds, position_ids, attention_mask)


class ParallelTransformerLayerPipe(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(self, args):
        hidden_states, position_ids, mask = args
        attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        outputs = LlamaDecoderLayer.forward(self,
                                            hidden_states,
                                            attention_mask,
                                            position_ids,
        )
        return (outputs[0], position_ids, mask)


class LayerNormPipe(LlamaRMSNorm):
    def forward(self, args):
        hidden_states, *_ = args
        last_hidden_states = super().forward(hidden_states)
        return (last_hidden_states,)


class LMLayerPipe(torch.nn.Linear):
    def forward(self, args):
        hidden_states, = args
        logits = super().forward(hidden_states)
        return (logits,)


def loss_fn(outputs, labels):
    # unpack
    logits, = outputs
    # all labels are `ignore_index` will cause nan
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
    )


def get_model(model_config: LlamaConfig, args, activation_checkpointing_config=None, **kwargs):
    class GPT2ModelPipe(PipelineModule):
        def __init__(self, model_config, **kwargs):
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
                layers=[
                    LayerSpec(EmbeddingPipe, model_config.vocab_size, model_config.hidden_size),
                    *[LayerSpec(ParallelTransformerLayerPipe, model_config)
                        for _ in range(model_config.num_hidden_layers)],
                    LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
                    LayerSpec(LMLayerPipe, model_config.hidden_size, model_config.vocab_size, bias=False),
                ],
                activation_checkpoint_interval=(1 if activation_checkpointing_config else 0),
                checkpointable_layers=["ParallelTransformerLayerPipe"],
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

    return GPT2ModelPipe(model_config,
                         loss_fn=loss_fn,
                         topology=topo,
                         base_seed=args.seed,
                         **kwargs)
