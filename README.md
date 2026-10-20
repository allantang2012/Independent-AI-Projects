# Independent AI Projects

A collection of independent, hands-on projects exploring deep learning, large language models, transformer architectures, and agentic AI systems.

These projects are developed outside formal coursework to strengthen my understanding of AI model development, training, fine-tuning, and evaluation.

## Projects

### 1. Decoder-Only Transformer from Scratch

**Technologies:** Python, PyTorch, Apple Silicon GPU (MPS)

Implemented and trained a GPT-style decoder-only transformer from scratch on the Tiny Shakespeare dataset.

**Architecture**
- 816,705 trainable parameters
- 4 transformer decoder blocks
- 4 causal attention heads per block
- 128-dimensional token embeddings
- 64-character context window

**Training Results**

| Training Steps | Training Loss | Validation Loss |
|---|---:|---:|
| 300 | 2.4652 | 2.4785 |
| 3,000 | 1.7159 | 1.8692 |
| 10,000 | 1.4297 | 1.6395 |

**Key Accomplishments**
- Implemented causal self-attention, multi-head attention, residual connections, layer normalization, and feedforward networks.
- Trained on approximately 1 million character tokens.
- Achieved approximately 62% lower validation loss relative to the initial untrained baseline.
- Implemented autoregressive text generation and model checkpoint loading.

**Project:** [decoder-transformer](./decoder-transformer)

### 2. DistilGPT2 Fine-Tuning

**Technologies:** Python, PyTorch, Hugging Face Transformers, Apple Silicon GPU (MPS)

Fine-tuned an 81.9-million-parameter pretrained DistilGPT2 model on Shakespeare text using the causal language modeling objective.

**Fine-Tuning Configuration**
- 81,912,576 model parameters
- 304,222 training tokens using GPT-2 tokenization
- 50 optimizer steps
- Learning rate: 2e-5
- Batch size: 2
- Sequence length: 128

**Evaluation Results**

Measured on 100 fixed validation sequences.

| Metric | Pretrained | Fine-Tuned | Improvement |
|---|---:|---:|---:|
| Validation loss | 4.5429 | 4.1318 | 9.1% reduction |
| Perplexity | 93.97 | 62.29 | 33.7% reduction |

**Key Accomplishments**
- Loaded pretrained GPT-2-family model weights and tokenizer.
- Fine-tuned the model using next-token prediction.
- Evaluated pretrained and fine-tuned models on identical held-out sequences.
- Saved and reloaded model weights for independent text generation.

**Project:** [gpt2-fine-tuning](./gpt2-fine-tuning)

### 3. Multi-Agent AI System

**Status:** Planned

A future project exploring LLM-powered agent communication, task delegation, tool calling, and multi-agent orchestration.

## Project Goals

- Build a deeper understanding of AI architectures through hands-on implementation.
- Explore the differences between training models from scratch and adapting pretrained models.
- Develop reproducible experiments and quantitative evaluation methods.
- Apply these skills to increasingly complex agentic AI systems.

## Notes

The decoder projects use different tokenization methods and model architectures, so their raw language modeling losses are not directly comparable. The reported fine-tuning improvements refer to evaluation on held-out Tiny Shakespeare sequences, not general language-model benchmarks.