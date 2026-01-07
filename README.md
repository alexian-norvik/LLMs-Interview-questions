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

## 13. What is a hallucination in LLMs?
Hallucination refers to the generation of plausible but incorrect or fictional information. For example, the model might invent citations, fake statistics, or historical facts. It's one of the most studied failure modes and is especially problematic in high-stakes applications like mediciine or law.

## 14. Are LLMs deterministic in their responses?
Not inherently. LLMs use probabilistic sampling methods (e.g., top-k, nucleus sampling) when generating text. This means responses can vary across runs with the same input. However, by setting temperature to 0 and using greedy decoding, you can force deterministic outputs.

## 15. What is the temperature setting in LLMs?
Temperature controls randomness in output generation:
* Low temperature (e.g., 0.2): More focused, deterministic outputs.
* High temperature (e.g., 0.9): More diverse and creative, but less predictable.

It's a trade-off between accuracy and creativity, useful in both business and creative contexts.

## 16. How does transfer learning apply to LLMs?
LLMs use transfer learning by leveraging generalized pretraining knowledge and adapting it to specific downstream tasks through fine-tuning or prompt-based learning. This significantly reduces the data and compute needed for new applications.

## 17. What is pretraining and how does it differ from fine-tuning?
Pretraining is the large-scale unsupervised learning phase where the model learns general language structure and world knowledge. Fine-tuning comes afterward and is task-specific. Together, they form the two-stage pipeline that powers most state-of-the-art LLMs today.

## 18. What is the role of attention in LLMs?
Attention mechanisms allow the model to weigh the importance of each word relative to others in a sentence. For example, in the phrase "The thropy didn't fit in the suitcase because it was too small", attention helps resolve the reference of "it". This is central to understanding context and reasoning.

## 19. How do LLMs differ from traditional NLP models?
Traditional NLP approaches were task-specific and relied heavily on feature engineering and labeled data. LLMs are end-to-end models that learn representations directly from raw text, offering much better generalization and fewer domain constraints. They're also far more scalable and flexible across tasks.

## 20. Can LLMs understand non-textual data like images or audio?
Base LLMs are text-only. However, multimodal models like GPT-4o, Gemini, and Claude 3 Opus integrate vision, audio, and even video. These models can process PDFs, screenshots, diagrams, or spoken language-expanding the scope of LLM applications beyond traditional NLP.
