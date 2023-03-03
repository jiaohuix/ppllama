# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import paddle

from ppllama.tokenizer import Tokenizer
from ppllama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = paddle.full((bsz, total_len), fill_value=self.tokenizer.pad_id,dtype="int64")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = paddle.to_tensor(t,dtype="int64")
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = paddle.nn.functional.softmax(logits / temperature, axis=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = paddle.argmax(logits, axis=-1)
            next_token = next_token.reshape([-1])
            # only replace token if prompt has already been generated
            next_token = paddle.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort = paddle.sort(probs, axis=-1, descending=True)
    probs_idx = paddle.argsort(probs, axis=-1, descending=True)
    probs_sum = paddle.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort  = probs_sort / (probs_sort.sum(axis=-1, keepdim=True))
    next_token = paddle.multinomial(probs_sort, num_samples=1)
    next_token = gather(probs_idx, -1, next_token)

    # test torch multinomial
    # next_token = torch.multinomial(torch.from_numpy(probs_sort.numpy()), num_samples=1)
    # next_token = torch.gather(torch.from_numpy(probs_idx.numpy()), -1, next_token)

    return next_token

def gather(x, axis, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if axis < 0: # last dim
        axis = x.ndim + axis
    nd_index = []
    for k in range(x.ndim):
        if k == axis:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * x.ndim
            reshape_shape[k] = x.shape[k]
            dim_index = paddle.expand(paddle.arange(x.shape[k], dtype=index.dtype).reshape(reshape_shape),
                                      index_shape).flatten()
            nd_index.append(dim_index)
    paddle_out = paddle.gather_nd(x, paddle.stack(nd_index, axis=-1)).reshape(index_shape)
    return paddle_out


if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)
    # torch.manual_seed(1)
    paddle.seed(1)
    bsz,seq,vocab = 4,10,100
    p=0.8
    logits = np.random.randn(bsz,vocab)
    logits = paddle.to_tensor(logits)
    temperature=1
    probs = paddle.nn.functional.softmax(logits / temperature, axis=-1)
    next_token = sample_top_p(probs,p)
    print(next_token)