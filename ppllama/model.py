# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import paddle
from paddle import nn
import paddle.nn.functional as F
import paddle.nn.initializer as I

import paddle.distributed as dist


import paddle.distributed as dist
from paddle.distributed.fleet.meta_parallel import (
    VocabParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear
)
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024


class RMSNorm(paddle.nn.Layer):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(shape=[dim],dtype="float32",default_initializer=I.Constant(1.))

    def _norm(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x).astype(dtype=x.dtype)
        return output * self.weight

def paddle_polar(r, theta):
    real = r * paddle.cos(theta)
    imag = r * paddle.sin(theta)
    return paddle.complex(real, imag)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (paddle.arange(0, dim, 2)[: (dim // 2)].astype(dtype="float32") / dim))
    t = paddle.arange(end).astype(dtype="float32")  # type: ignore
    freqs = paddle.outer(t, freqs).astype(dtype="float32") # type: ignore  # 外积
    freqs_cis = paddle_polar(paddle.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: paddle.Tensor, x: paddle.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == [x.shape[1], x.shape[-1]]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(shape)


def apply_rotary_emb(
    xq: paddle.Tensor,
    xk: paddle.Tensor,
    freqs_cis: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    xq_ = paddle.as_complex(xq.astype(dtype="float32").reshape([*xq.shape[:-1], -1, 2]))
    xk_ = paddle.as_complex(xk.astype(dtype="float32").reshape([*xk.shape[:-1], -1, 2]))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = paddle.as_real(xq_ * freqs_cis).flatten(3)
    xk_out = paddle.as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


class Attention(nn.Layer):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // dist.get_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            has_bias=False,
            gather_output=False,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            has_bias=False,
            gather_output=False,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            has_bias=False,
            gather_output=False,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
        )

        self.cache_k = paddle.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )
        self.cache_v = paddle.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )

    def forward(self, x: paddle.Tensor, start_pos: int, freqs_cis: paddle.Tensor, mask: Optional[paddle.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape([bsz, seqlen, self.n_local_heads, self.head_dim])
        xk = xk.reshape([bsz, seqlen, self.n_local_heads, self.head_dim])
        xv = xv.reshape([bsz, seqlen, self.n_local_heads, self.head_dim])

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)


        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose([0,2,1,3])
        keys = keys.transpose([0,2,1,3])
        values = values.transpose([0,2,1,3])
        scores = paddle.matmul(xq, keys.transpose([0,1,3,2])) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.astype("float32"), axis=-1).astype(xq.dtype)
        output = paddle.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            [0,2,1,3]
        ).reshape([bsz, seqlen, -1])

        return self.wo(output)


class FeedForward(nn.Layer):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            # dim, hidden_dim, has_bias=False, gather_output=False, init_method=lambda x: x
            dim, hidden_dim, has_bias=False, gather_output=False
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, has_bias=False, input_is_parallel=True
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, has_bias=False, gather_output=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Layer):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: paddle.Tensor, start_pos: int, freqs_cis: paddle.Tensor, mask: Optional[paddle.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Layer):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim
        )

        self.layers = paddle.nn.LayerList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, has_bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @paddle.no_grad()
    def forward(self, tokens: paddle.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = paddle.full((1, 1, seqlen, seqlen), float("-inf"))
            mask = paddle.triu(mask, diagonal=start_pos + 1).astype(h.dtype) # cast

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.astype(dtype="float32")


if __name__ == '__main__':
    import numpy as np
    from reprod_log import ReprodLogger
    reprod_logger = ReprodLogger()

    np.random.seed(1)
    # torch.manual_seed(1)
    paddle.seed(1)
    paddle.set_device("cpu")


    # test1: RMSnorm
    # dim = 8
    # norm = RMSNorm(dim=dim)
    # x = paddle.to_tensor(np.random.randn(2,dim).astype("float32"))
    # o = norm(x)
    # print(o)

    # test2:
    bsz,seqlen = 4, 16
    dim = 768
    n_heads = 16
    word_size = 1
    n_local_heads = n_heads // word_size # 16
    head_dim = dim // n_heads
    max_seq_len=1024
    freqs_cis = precompute_freqs_cis(
        dim //n_heads, max_seq_len * 2
    )

    # test3: apply_rotary_emb

    start_pos = 0
    freqs_cis = freqs_cis[start_pos: start_pos + seqlen]
    reprod_logger.add("freqs_cis", freqs_cis.cpu().detach().numpy())

    xq = paddle.to_tensor(np.random.randn(bsz,seqlen,dim).astype("float32"))
    xk = paddle.to_tensor(np.random.randn(bsz,seqlen,dim).astype("float32"))
    xq = xq.reshape([bsz, seqlen, n_local_heads, head_dim])
    xk = xk.reshape([bsz, seqlen, n_local_heads, head_dim])
    reprod_logger.add("xq_raw", xq.cpu().detach().numpy())
    reprod_logger.add("xk_raw", xk.cpu().detach().numpy())
    # RuntimeError: The size of tensor a (16) must match the size of tensor b (15) at non-singleton dimension 2

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    reprod_logger.add("xq", xq.cpu().detach().numpy())
    reprod_logger.add("xk", xk.cpu().detach().numpy())
    reprod_logger.save("../forward_paddle.npy")
