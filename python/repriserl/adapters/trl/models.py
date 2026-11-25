"""Shared model, prompt and exact arithmetic reward for the stock and replay trainers."""
from repriserl.config import RunConfig
from repriserl.verifiers.number_expression import Problem, verify_expression


def problem_prompt(problem: Problem, tokenizer=None) -> str:
    numbers = " ".join(map(str, problem.numbers))
    prompt = (f"Use each number exactly once: {numbers}. Make {problem.target} "
            "using + - * / and parentheses. Return only the expression.\nAnswer:")
    if tokenizer is not None and tokenizer.chat_template:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
    return prompt


def reward_completions(completions, numbers, target, **kwargs) -> list[float]:
    return [
        verify_expression(Problem(values, goal), completion).reward
        for completion, values, goal in zip(completions, numbers, target, strict=True)
    ]


def load_model_and_tokenizer(config: RunConfig, *, tiny: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(config.seed)
    if not tiny:
        if not config.model_revision:
            raise ValueError("an immutable model_revision is required for a Hub model")
        from repriserl.gpu_run import check_device
        check_device(config.allow_shared_gpu)
        if torch.cuda.device_count() != 1:
            raise RuntimeError("the trainer requires exactly one visible CUDA device")
        kwargs = dict(revision=config.model_revision, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(config.model, **kwargs)
        # Keep optimizer master parameters in FP32. Trainer autocasts the forward pass.
        model = AutoModelForCausalLM.from_pretrained(config.model, dtype=torch.float32, **kwargs)
    else:
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        tokens = ["<pad>", "<eos>", "<unk>", "+", "-", "*", "/", "(", ")",
                  ".", ":", "Use", "each", "number", "exactly", "once", "Make",
                  "using", "and", "parentheses", "Return", "only", "the", "expression", "Answer"]
        tokens.extend(str(value) for value in range(1, 21))
        backend = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                             eos_token="<eos>", unk_token="<unk>")
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokenizer), n_layer=1, n_head=2, n_embd=16,
            n_positions=config.prompt_token_cap + config.completion_token_cap,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.eos_token_id,
        ))
    if tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer requires an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def enable_tiny_cpu_scores(model):
    """Compute TRL's requested head outputs on CPU, where its Triton kernel cannot run."""
    from functools import wraps
    from trl.trainer.utils import FusedCausalLMOutput
    if model.config.model_type != "gpt2" or model.device.type != "cpu":
        raise ValueError("the dense fixture scorer requires a CPU GPT-2 model")
    original = model.forward

    @wraps(original)
    def forward(*args, fused_lm_head=False, labels=None, **kwargs):
        if not fused_lm_head:
            return original(*args, labels=labels, **kwargs)
        attention = kwargs.get("attention_mask")
        if attention is not None and "position_ids" not in kwargs:
            # Match generation positions for left-padded GPT-2 prompts.
            kwargs["position_ids"] = (attention.long().cumsum(-1) - 1).masked_fill(attention == 0, 0)
        kwargs["use_cache"] = False
        logits = original(*args, **kwargs).logits[:, :-1]
        labels = labels[:, 1:]
        active = labels != -100
        logp = logits.log_softmax(-1)
        selected = logp.gather(-1, labels.masked_fill(~active, 0).unsqueeze(-1)).squeeze(-1)
        entropy = -(logp.exp() * logp).sum(-1)
        return FusedCausalLMOutput(log_probs=selected.masked_fill(~active, 0.0),
                                   entropy=entropy.masked_fill(~active, 0.0), label_mask=active)
    model.forward = forward
