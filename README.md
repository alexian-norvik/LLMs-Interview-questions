# LLMs-Interview-questions

## 1. What is a Large Language Model (LLM)?
A Large Language Model is a type of deep learning model designed to understand, generate, and manipulate human language. LLMs are typically based on Transformer architectures and are trained on massive corpora to learn linguistic patterns, world knowledge, and reasoning abilities. They underpin many applications, from chatbots to code generation tools.

## 2. How are LLMs trained?
LLMs undergo pretraining on vase, diverse datasets using self-supervised learning. The most common objective is next-token prediction: given a sequence of tokens, the model predicts what comes next. This allows LLMs to learn grammar, semantics, and even factual knowledge without labeled datasets. Training requires significant computational resources and optimization techniques like AdamW and learning rate scheduling.

## 3. What is a token in LLMs?
In LLMs, a token is the smallest unit of text the model processes. Tokens can be full words, subwords, or even characters depending on the tokenizer (e.g., Byte-Pair Encoding or SentencePiece). For instance, the word "engineering" might be split into "engine" and "eering". Tokenization ensures that the model can handle rare or compound words efficiently.

## 4. What is a context window and why does it matter?
The context window defines how many tokens the model can attend to at one time. A model like GPT-3 has a context limit of 2,048 tokens, while newer models like GPT-4 and Claude 3 go up to 100,000+. A large context window enables better handling of long documents, persistent dialogue, and multi-part reasoning. Context limitations directly affect model memory and coherence.

## 5. What are embedding layers and why are they important?
Embedding layers are the entry point of LLMs. They map discrete tokens into high-dimensional continuous vectors, allowing the model to capture syntactic and semantic relationships. Words with similar meanings, like "car", and "vehicle", will have embeddings close in vector space, which improves the model's ability to generalize language.
