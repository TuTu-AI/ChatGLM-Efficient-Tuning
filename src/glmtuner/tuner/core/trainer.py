import os
import torch
from typing import Dict, Optional
import psutil
import time

from transformers import Seq2SeqTrainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from peft import PeftModel

from glmtuner.extras.constants import FINETUNING_ARGS_NAME, VALUE_HEAD_FILE_NAME
from glmtuner.extras.logging import get_logger
from glmtuner.extras.save_and_load import get_state_dict, load_trainable_params, load_valuehead_params
from glmtuner.hparams import FinetuningArguments


logger = get_logger(__name__)


class PeftTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
    """

    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self._remove_log()

    def _remove_log(self):
        if self.is_world_process_zero() and os.path.exists(os.path.join(self.args.output_dir, "trainer_log.jsonl")):
            logger.warning("Previous log file in this folder will be deleted.")
            os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoint.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_size = sum(tensor.numel() * tensor.element_size() for tensor in self.model.parameters())
        
        while True:
            available_ram = psutil.virtual_memory().available
            if available_ram > model_size * 2:
        
                logger.info(f"Saving model checkpoint to {output_dir}")
                model = unwrap_model(self.model)
                rank = torch.distributed.get_rank()
                if hasattr(model, "pretrained_model"): # for models with valuehead (currently using LoRA only)
                    print(f'Saving pretrained model for {rank}...')
                    backbone_model = getattr(model, "pretrained_model")
                    torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
                else:
                    print(f'Not saving pretrained model for {rank}')
                    backbone_model = model

                if isinstance(backbone_model, PeftModel): # LoRA tuning
                    print(f'saving peft model for {rank}...')
                    backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))
                    print('Saving peft model done')
                elif isinstance(backbone_model, PreTrainedModel): # freeze/full-tuning or p_tuning
                    print(f'saving pretrained model for {rank}...')
                    backbone_model.config.use_cache = True
                    backbone_model.save_pretrained(
                        output_dir,
                        state_dict=get_state_dict(backbone_model, trainable_only=(self.finetuning_args.finetuning_type != "full")),
                        safe_serialization=self.args.save_safetensors
                    )
                    backbone_model.config.use_cache = False
                    if self.tokenizer is not None:
                        self.tokenizer.save_pretrained(output_dir)
                else:
                    logger.warning("No model to save.")

                with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
                    f.write(self.args.to_json_string() + "\n")
                self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))
                break
            else:
                logger.warning(f"Not enough RAM to save model checkpoint to {output_dir}. Retrying in 60 seconds...")
                time.sleep(15)    

    def _load_best_model(self):
        r"""
        Loads trainable parameters from model checkpoint.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")

        model = unwrap_model(self.model)
        backbone_model = getattr(model, "pretrained_model") if hasattr(model, "pretrained_model") else model

        if isinstance(backbone_model, PeftModel):
            backbone_model.load_adapter(self.state.best_model_checkpoint, backbone_model.active_adapter)
            if hasattr(model, "v_head") and load_valuehead_params(model, self.state.best_model_checkpoint):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })
        else: # freeze/full-tuning or p_tuning
            load_trainable_params(backbone_model, self.state.best_model_checkpoint)
