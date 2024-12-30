# tinygpt-2 : A GPT-2 architecture from Scratch

Welcome! This repository is all about understanding and building a GPT-2 model from scratch using PyTorch. Inspired by Andrej Karpathy's brilliant tutorials, this project breaks down the process into simple steps while keeping it engaging and accessible.

### Cool Features

- **Customizable Settings**: Tweak the model's vocabulary size, embedding dimensions, layers, and more with the `GPTConfig` class.
- **Attention Mechanisms**: Implements causal self-attention to ensure predictions only consider past tokens—think of it as the model's memory.
- **Streamlined Layers**: Efficient MLP layers make computation a breeze.
- **Pretrained Model Support**: Plug in Hugging Face models for quick results.
- **Text Creation**: Use the `generate` method to produce coherent and creative text.

## Getting Started

First, make sure you have Python 3.7 or later. Then, install the required libraries:

```bash
pip install torch tiktoken
```

### How to Use

#### Step 1: Set Up Your Model

```python
from model import GPT, GPTConfig

# Configure your model
config = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    bias=True
)

# Initialize your GPT model
model = GPT(config)
```

#### Step 2: Load Pretrained Weights

```python
from transformers import GPT2LMHeadModel

# Load pretrained GPT-2 weights
pretrained_model = GPT.from_pretrained("gpt2")
```

### Examples to Try

#### Example 1: Finish the Sentence

```python
prompt = "The universe is vast and"
tokens = enc.encode(prompt)
inputs = torch.tensor([tokens], dtype=torch.long)

# Generate the next part of the text
generated = model.generate(inputs, max_new_tokens=15)
print(enc.decode(generated[0].tolist()))
```

#### Example 2: Write a Short Story

```python
prompt = "Once upon a time in a land far, far away,"
tokens = enc.encode(prompt)
inputs = torch.tensor([tokens], dtype=torch.long)

# Let the model continue the story
generated = model.generate(inputs, max_new_tokens=20)
print(enc.decode(generated[0].tolist()))
```

### Shoutout

Big thanks to Andrej Karpathy for his incredible tutorials and code snippets that made this project possible. His work continues to inspire the AI community.

## License

This project is licensed under the MIT License.&#x20;

