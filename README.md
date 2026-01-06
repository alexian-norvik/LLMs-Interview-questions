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

## 6. How do LLMs handle out-of-vocabulary (OOV) words?
Unlike earlier NLP systems with fixed vocabularies, modern LLMs mitigate OOV issues via subword tokenization. Unknown words are decomposed into known sub-units. For example, "bioinformatics" could be tokenized as "bio", "inform", "atics". This strategy ensures that the model can interpret new or rare words contextually.

## 7. What are the most common real-world use cases for LLMs?
LLMs power a broad spectrum of AI applications, including:
* Conversational AI (chatbots, voice assistants)
* Content creation (blogs, ads, summaries)
* Code generation & completion (GitHub Copilot)
* Information retrieval and Q&A
* Sentiment analysis and moderation
* Language Transalation and localization

## 8. What is fine-tuning and why is it used?
Fine-tuning is the process of continuing training on a smaller, domain-specific dataset after general pretraining. It allows LLMs to specialize in a particular industry (e.g., legal, healthcare, finance) or task (e.g., summarization, classification). It improves performance, reduce hallucination, and enables alignment with brand or organizational voice.

## 9. What is prompt engineering?
Prompt engineering involves crafting precise input prompts that guide the model toward a desired output. It's critical when using models in a zero-shot or few-shot setting. A well-engineered prompt can significantly improve model performance without requiring retraining-making it a high-leverage skill for AI practitioners.

## 10. What is zero-shot learning in the context of LLMs?
Zero-shot learning refers to the model's ability to perform tasks without prior examples. For instance, asking in LLM, "Translate 'good morning' into Japanese" assumes the model can infer the instruction and complete the task without training examples. This showcases the model's inherent generalization capability.

## 11. What is few-shot learning and how does it differ from zero-shot?
Few-shot learning involves giving the model a small numer of task-specific examples within the prompt. This primes the model for the expected structure and output style. For instance, few-shot prompting is particularly effecitve in classification, summarization, or role-based dialogue simulations.

## 12. What are the main challenges of deploying LLMs in production?
key challenges include:
* Cost: Serving large language models requires GPUs or TPUs.
* Latency: Response time can be high for large inputs.
* Hallucination: LLMs may confidently generate incorrect information.
* Bias: Models may reflect societal or dataset biases.
* Data privacy: Sensitive inputs need to be managed carefully.
