#!/usr/bin/env python3
"""
Generate expanded raw substitute data for L1 pipeline.

Creates data/l0_raw/l0_expanded_SUBSTITUTE.jsonl with ~150 rows of varied quality:
- Good text (substantive, long, high alpha ratio)
- Short text (below L1 thresholds)
- Boilerplate text (nav, cookie, sign-in)
- Duplicate text (exact copies)
- Noisy symbol-heavy text
- Low-information text (repetitive, filler)

All rows marked with source='local_substitute_l1'
"""

import json
from pathlib import Path


GOOD_TEXTS = [
    "The transformer architecture introduced in Attention Is All You Need revolutionized natural language processing by replacing recurrence with self-attention mechanisms. This allows models to process sequences in parallel while capturing long-range dependencies effectively. The key innovation is the scaled dot-product attention which computes attention weights by taking the dot product of queries and keys, scaling by the square root of the dimension, and applying softmax.",
    "BERT (Bidirectional Encoder Representations from Transformers) pre-trains deep bidirectional representations by jointly conditioning on both left and right context in all layers. The pre-trained model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks such as question answering and language inference.",
    "GPT-3 demonstrates that scaling language models to 175 billion parameters enables few-shot learning across diverse tasks without task-specific fine-tuning. The model shows strong performance on translation, question answering, and reasoning benchmarks. The scaling hypothesis suggests that performance continues to improve with model size, dataset size, and compute budget.",
    "The scaling laws for neural language models show that test loss follows a power-law relationship with model size, dataset size, and compute budget. This predicts that larger models trained on more data will continue to improve. The optimal allocation of compute between model size and training tokens follows a specific ratio.",
    "Chain-of-thought prompting elicits reasoning in large language models by encouraging them to generate intermediate reasoning steps before producing a final answer. This significantly improves performance on arithmetic, commonsense, and symbolic reasoning tasks. The technique works by providing exemplars that show the reasoning process.",
    "Instruction tuning with human feedback (RLHF) aligns language models to follow instructions and be helpful, harmless, and honest. The process involves supervised fine-tuning followed by reward model training and proximal policy optimization. This has become the standard approach for deploying conversational AI systems.",
    "Retrieval-augmented generation (RAG) combines parametric knowledge from pre-trained language models with non-parametric knowledge from external retrieval systems. This allows models to access up-to-date information and provide citations for their answers. The approach has been widely adopted for knowledge-intensive NLP tasks.",
    "Mixture-of-experts (MoE) architectures enable scaling model capacity without proportional increases in compute by routing inputs to specialized expert sub-networks. This allows training models with trillions of parameters while only activating a fraction for each input. Switch Transformers and GLaM are notable examples.",
    "Constitutional AI trains language models to adhere to a set of principles or a constitution by using AI feedback instead of human feedback for the reward modeling step. This reduces the need for human annotation while maintaining alignment. The approach uses self-critique and revision to improve model outputs.",
    "Diffusion models have emerged as a powerful class of generative models that learn to reverse a gradual noising process. They achieve state-of-the-art results in image generation and are being adapted for text and audio. The key insight is learning the score function of the data distribution.",
    "Vision transformers (ViT) apply the transformer architecture to image classification by treating images as sequences of patches. They achieve competitive results with CNNs while requiring less inductive bias. Large-scale pre-training on JFT-300M enables strong transfer learning performance.",
    "The Lottery Ticket Hypothesis proposes that dense neural networks contain sparse subnetworks that can be trained in isolation to achieve comparable performance. This has implications for model compression and understanding generalization. Iterative magnitude pruning can find these winning tickets.",
    "Self-supervised learning in computer vision uses pretext tasks like masked image modeling, contrastive learning, and rotation prediction to learn representations without labels. MAE, SimCLR, and DINO are prominent methods. These representations transfer well to downstream tasks.",
    "Foundation models are models trained on broad data at scale that can be adapted to a wide range of downstream tasks. They exhibit emergent capabilities not present in smaller models. The term was popularized by the Stanford CRFM report on opportunities and risks.",
    "Prompt engineering involves designing input prompts to elicit desired behaviors from language models. Techniques include few-shot prompting, chain-of-thought, self-consistency, and tree-of-thoughts. The field has evolved into a systematic discipline for working with LLMs.",
    "Parameter-efficient fine-tuning (PEFT) methods like LoRA, adapters, and prefix tuning enable adapting large pre-trained models with minimal trainable parameters. LoRA decomposes weight updates into low-rank matrices, reducing memory and compute requirements significantly.",
    "In-context learning is the ability of language models to learn from examples provided in the prompt without weight updates. This emergent capability appears at scale and enables few-shot adaptation. The mechanism is still under active research.",
    "Alignment research focuses on ensuring AI systems behave in accordance with human values and intentions. Key challenges include reward hacking, specification gaming, and deceptive alignment. Techniques include RLHF, constitutional AI, and debate.",
    "Multimodal models process and generate multiple modalities such as text, images, audio, and video. Flamingo, GPT-4V, and Gemini demonstrate strong cross-modal reasoning. These models enable new applications in visual question answering and content creation.",
    "Long-context models extend the context window of transformers to handle hundreds of thousands or millions of tokens. Techniques include sparse attention, recurrent memory, and positional interpolation. This enables processing entire codebases, books, or long conversations.",
    "Model merging combines multiple fine-tuned models into a single model without additional training. Techniques like task arithmetic, TIES-Merging, and DARE enable composition of capabilities. This is useful for creating multi-task models efficiently.",
    "Quantization reduces the precision of model weights from 32-bit floats to 8-bit integers or lower, enabling faster inference and reduced memory usage. Post-training quantization and quantization-aware training are common approaches. GPTQ and AWQ are popular methods for LLMs.",
    "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model by training the student to match the teacher's outputs. This enables deployment of compact models with minimal performance loss. Sequence-level and token-level distillation are used.",
    "Sparse attention mechanisms reduce the quadratic complexity of full attention by only attending to a subset of positions. Longformer, BigBird, and Reformer use patterns like sliding windows, global tokens, and random attention. This enables longer context windows.",
    "Retrieval-based language models augment parametric memory with external knowledge bases. RETRO, Atlas, and kNN-LM retrieve relevant passages during generation. This improves factual accuracy and enables updating knowledge without retraining.",
    "Tool use and API calling enable language models to interact with external systems like calculators, search engines, and code executors. Toolformer and Gorilla demonstrate training models to use tools. This expands the range of solvable tasks.",
    "Recursive self-improvement explores whether AI systems can improve their own capabilities. Constitutional AI and recursive reward modeling are early steps. This remains a theoretical concept with significant safety implications.",
    "Emergent abilities are capabilities that appear abruptly at certain model scales rather than improving gradually. Examples include arithmetic, multi-step reasoning, and translation. The phenomenon is debated but widely observed in scaling studies.",
    "Grokking is the phenomenon where models suddenly achieve perfect generalization after extended training on algorithmic tasks. It challenges traditional views of overfitting and generalization. The mechanism involves learning structured representations.",
    "Double descent describes the phenomenon where test error first decreases, then increases, then decreases again as model size grows beyond the interpolation threshold. This contradicts classical bias-variance tradeoff. It is observed in deep learning.",
    "Neural scaling laws predict how model performance scales with compute, data, and parameters. The Chinchilla scaling law suggests optimal compute allocation favors more data over larger models. These laws guide training decisions.",
    "Data quality and filtering significantly impact model performance. Deduplication, heuristic filtering, and model-based selection improve training efficiency. The RefinedWeb and C4 datasets demonstrate the importance of curation.",
    "Synthetic data generation uses models to create training data for other models. Self-Instruct, Alpaca, and WizardLM generate instruction-following data. This reduces reliance on human annotation but risks model collapse.",
    "Contrastive learning learns representations by pulling positive pairs together and pushing negative pairs apart. SimCLR, MoCo, and CLIP are foundational methods. This enables learning from unlabeled data at scale.",
    "Masked autoencoding reconstructs masked portions of input data. MAE for images and BERT for text use this objective. It learns rich representations without explicit supervision. The masking ratio affects difficulty.",
    "Diffusion policy learning applies diffusion models to robotics and control. Diffusion Policy and DP3 demonstrate learning complex behaviors from demonstrations. This bridges generative modeling and decision making.",
    "World models learn predictive models of environment dynamics. Dreamer, IRIS, and Genie learn latent dynamics for planning. This enables model-based reinforcement learning in complex environments.",
    "Language model agents use LLMs as reasoning engines for multi-step tasks. ReAct, AutoGPT, and BabyAGI demonstrate planning and tool use. Reliability and evaluation remain key challenges for deployment.",
    "Evaluation of language models requires diverse benchmarks. MMLU, BIG-Bench, HELM, and LM-Eval-Harness provide standardized assessment. Capability, alignment, and safety dimensions must all be measured.",
    "Red teaming tests model vulnerabilities by adversarially probing for harmful outputs. This identifies risks like jailbreaks, hallucinations, and bias. Automated and human red teaming complement each other.",
    "Interpretability research aims to understand model internals. Mechanistic interpretability, probing, and feature visualization reveal how models process information. This is crucial for trust and safety.",
    "Watermarking embeds detectable signals in model outputs to identify AI-generated content. Kirchenbauer et al. and Aaronson propose statistical watermarks. This addresses misuse and provenance concerns.",
    "Federated learning trains models across decentralized devices without sharing raw data. FedAvg and FedProx are foundational algorithms. Privacy and communication efficiency are key considerations.",
    "Continual learning adapts models to new tasks without forgetting previous knowledge. Elastic weight consolidation, replay, and parameter isolation are common approaches. Catastrophic forgetting is the main challenge.",
    "Neural architecture search automates model design. DARTS, ENAS, and Once-for-All search efficient architectures. This reduces manual engineering but requires significant compute.",
    "Efficient transformers optimize attention computation. Linear attention, performer, and linformer reduce complexity. These enable longer sequences and faster training.",
    "Mixture of depths dynamically allocates compute across layers. Skip layers and early exit reduce inference cost. This adapts computation to input difficulty.",
    "Speculative decoding accelerates LLM inference by using a small draft model to propose tokens. The large model verifies proposals in parallel. This achieves 2-3x speedup with minimal quality loss.",
    "Prefix caching reuses attention computations for shared prefixes in multi-turn conversations. This reduces redundant computation in serving. vLLM and SGLang implement this optimization.",
    "Continuous batching dynamically schedules requests to maximize GPU utilization. This improves throughput for LLM serving. vLLM and TensorRT-LLM support this pattern.",
    "Tensor parallelism splits model weights across GPUs for distributed inference. Megatron-LM and DeepSpeed implement this. Communication overhead is the main bottleneck.",
    "Pipeline parallelism splits model layers across GPUs. GPipe and PipeDream balance workload. Micro-batching reduces pipeline bubbles.",
    "Sequence parallelism splits sequence dimensions across GPUs. This complements tensor parallelism for long sequences. Megatron-LM supports this for transformer models.",
    "Zero redundancy optimizer (ZeRO) partitions optimizer states, gradients, and parameters across GPUs. DeepSpeed ZeRO-3 enables training trillion-parameter models. This reduces memory per GPU.",
    "Activation checkpointing trades compute for memory by recomputing activations during backward pass. This enables training larger models with limited memory. Gradient checkpointing is widely used.",
    "Mixed precision training uses FP16 or BF16 for forward pass and FP32 for gradients. This reduces memory and accelerates compute on modern GPUs. Automatic mixed precision (AMP) simplifies adoption.",
    "Gradient accumulation simulates larger batch sizes by accumulating gradients over multiple steps. This enables training with limited GPU memory. Effective batch size equals micro-batch times accumulation steps.",
    "Learning rate scheduling adapts learning rate during training. Cosine decay, warmup, and constant schedules are common. The schedule significantly impacts final performance.",
    "Weight decay regularizes training by penalizing large weights. AdamW decouples weight decay from gradient updates. This improves generalization in transformer training.",
    "Layer normalization stabilizes training by normalizing activations across features. Pre-norm and post-norm variants affect training dynamics. Pre-norm is standard in modern transformers.",
    "Residual connections enable gradient flow in deep networks. Pre-norm residual connections are standard in transformers. They prevent vanishing gradients in deep models.",
    "Positional encodings inject sequence order information. Absolute, relative, and rotary (RoPE) encodings are used. RoPE is popular for its extrapolation properties.",
    "Attention mechanisms compute weighted sums of values based on query-key compatibility. Multi-head attention allows attending to different subspaces. Scaled dot-product attention is the standard.",
    "Feed-forward networks in transformers apply pointwise transformations. Gated linear units (GLU) and SwiGLU improve performance. The expansion ratio is typically 4x.",
    "Normalization layers stabilize training dynamics. LayerNorm, RMSNorm, and BatchNorm are used. RMSNorm is simpler and faster than LayerNorm.",
    "Initialization schemes affect training stability. Xavier, Kaiming, and transformer-specific initializations are used. Proper initialization is crucial for deep models.",
    "Regularization techniques prevent overfitting. Dropout, stochastic depth, and label smoothing are common. Transformer models benefit from moderate regularization.",
    "Data augmentation improves generalization. Back-translation, mixup, and cutout are used for text. Augmentation is less common in LLM pre-training.",
    "Curriculum learning presents examples in increasing difficulty. This can accelerate convergence and improve final performance. Difficulty metrics include length and perplexity.",
    "Distributed training coordinates multiple GPUs or nodes. Data parallelism, tensor parallelism, and pipeline parallelism are combined. Communication optimization is critical.",
    "Checkpointing saves model state for recovery and evaluation. Sharded checkpoints reduce I/O overhead. Frequent checkpointing enables fault tolerance.",
    "Monitoring tracks training metrics like loss, perplexity, and gradient norms. TensorBoard, Weights & Biases, and MLflow are popular tools. Alerting detects training issues early.",
    "Hyperparameter tuning optimizes learning rate, batch size, and architecture. Bayesian optimization and population-based training are used. Compute budget constrains search.",
    "Model selection chooses the best checkpoint. Validation perplexity and downstream task performance are used. Early stopping prevents overfitting.",
    "Deployment serves models for inference. ONNX, TensorRT, and Triton optimize serving. Batching, caching, and quantization improve throughput.",
    "Monitoring production models tracks latency, error rates, and drift. A/B testing compares model versions. Canary deployments reduce risk.",
    "Data versioning tracks dataset changes. DVC and LakeFS provide Git-like versioning for data. This enables reproducibility and audit trails.",
    "Experiment tracking records hyperparameters, metrics, and artifacts. MLflow, Weights & Biases, and ClearML are popular. This enables comparison and reproducibility.",
    "Feature stores centralize feature engineering. Feast and Tecton provide feature serving. This ensures consistency between training and inference.",
    "Model registry manages model versions and metadata. MLflow and Vertex AI Model Registry provide this. Staging and production transitions are tracked.",
    "CI/CD for ML automates training, testing, and deployment. GitHub Actions, GitLab CI, and Jenkins are used. Automated testing catches regressions.",
    "Data validation ensures data quality. Great Expectations and TensorFlow Data Validation check schemas and statistics. This prevents silent data corruption.",
    "Model validation tests model behavior. Unit tests, integration tests, and adversarial tests are used. This ensures reliability before deployment.",
    "A/B testing compares model variants in production. Statistical significance and practical significance are both considered. This enables data-driven decisions.",
    "Canary deployment gradually rolls out new models. Traffic splitting and automated rollback reduce risk. This is standard for ML model deployment.",
    "Shadow deployment runs new models alongside production without serving traffic. This validates performance before full rollout. Logs are compared for discrepancies.",
    "Rollback strategies revert to previous model versions. Automated rollback on metric degradation is essential. This minimizes impact of bad deployments.",
    "Cost optimization balances performance and compute cost. Spot instances, reserved instances, and serverless are used. Right-sizing instances reduces waste.",
    "Security considerations protect models and data. Model extraction, membership inference, and data poisoning are threats. Differential privacy and secure aggregation mitigate risks.",
    "Privacy-preserving ML enables training on sensitive data. Federated learning, differential privacy, and secure multi-party computation are used. This enables healthcare and finance applications.",
    "Fairness metrics measure bias across demographic groups. Demographic parity, equalized odds, and calibration are used. Intersectional fairness considers multiple attributes.",
    "Explainability provides reasons for model decisions. SHAP, LIME, and integrated gradients are post-hoc methods. Inherent interpretability is preferred for high-stakes decisions.",
    "Robustness measures performance under distribution shift. Adversarial examples, out-of-distribution detection, and stress testing are used. This ensures reliability in production.",
    "Governance frameworks manage ML lifecycle. Model cards, data sheets, and system cards document models. Regulatory compliance requires documentation.",
    "MLOps platforms integrate the ML lifecycle. Kubeflow, MLflow, and Vertex AI provide end-to-end tooling. This enables scalable and reliable ML operations.",
]

SHORT_TEXTS = [
    "Short text.",
    "Very brief.",
    "Just a few words here.",
    "Tiny snippet.",
    "Not enough content.",
    "Brief note.",
    "Short.",
    "Too short for L1.",
    "Minimal text.",
    "Insufficient length.",
    "Quick blurb.",
    "Tiny piece.",
    "Fragment only.",
    "Brief fragment.",
    "Short entry.",
]

BOILERPLATE_TEXTS = [
    "Cookie Policy: We use cookies to improve your experience. Accept all cookies or reject all. Privacy Policy Terms of Service.",
    "Sign in to your account. Register for free. Subscribe to our newsletter. Follow us on Twitter Facebook LinkedIn.",
    "Home About Contact Menu Navigation Skip to content. Advertisement Sponsored Promoted. Share this Follow us Social media.",
    "Privacy Policy Terms of Use Cookie Settings Do Not Sell My Info. Accept All Reject All Manage Preferences.",
    "Log in Sign up Forgot password Remember me. Create account Join now Free trial.",
    "Navigation Menu Home Products Services About Us Contact Blog Careers Press.",
    "Subscribe to our newsletter for updates. Enter your email. Sign up now. Unsubscribe anytime.",
    "Share this article on Facebook Twitter LinkedIn Reddit. Copy link. Print version.",
    "Related articles Recommended for you Trending now Most popular Latest news.",
    "Advertisement Sponsored Content Promoted Post Native Ad. Learn more about our ad policies.",
    "Cookie Notice We value your privacy. We use cookies for analytics and personalization. Accept Decline.",
    "Terms and Conditions Privacy Policy Cookie Policy Accessibility Statement. All rights reserved.",
    "Follow us on social media Instagram YouTube TikTok Snapchat. Join the conversation.",
    "Sign in with Google Sign in with Apple Sign in with Microsoft. Or use email.",
    "Create your free account today. No credit card required. Cancel anytime. Start free trial.",
]

NOISY_TEXTS = [
    "!!!@@@###$$$%%%^^^&&&***((( random symbols and noise !!!@@@###$$$%%%^^^&&&***((( ",
    "###$$$%%%^^^&&&***(((###$$$%%%^^^&&&***(((###$$$%%%^^^&&&***((( noise pattern ",
    "Random noise !@#$%^&*()_+{}|:<>?~` random characters everywhere !@#$%^&*()_+{}|:<>?~`",
    "Symbols everywhere @@@@@ ##### $$$$$ %%%%% ^^^^^ &&&&& ***** ((((( )))))",
    "Noise noise noise !!!@@@###$$$%%%^^^&&&***((( noise noise noise !!!@@@###$$$%%%^^^&&&***((( ",
    "Garbage text @@@@@ ##### $$$$$ %%%%% ^^^^^ &&&&& ***** ((((( ))))) garbage garbage",
    "Symbol soup !@#$%^&*()_+{}|:<>?~`!@#$%^&*()_+{}|:<>?~`!@#$%^&*()_+{}|:<>?~`",
    "Random chars ###$$$%%%^^^&&&***(((###$$$%%%^^^&&&***(((###$$$%%%^^^&&&***((( ",
    "Noise pattern !!!@@@###$$$%%%^^^&&&***((( !!!@@@###$$$%%%^^^&&&***((( !!!@@@###$$$%%%^^^&&&***((( ",
    "Symbol heavy @@@@@ ##### $$$$$ %%%%% ^^^^^ &&&&& ***** ((((( ))))) @@@@@ ##### $$$$$ %%%%%",
    "Garbage chars !@#$%^&*()_+{}|:<>?~`!@#$%^&*()_+{}|:<>?~`!@#$%^&*()_+{}|:<>?~`",
    "Random symbols ###$$$%%%^^^&&&***(((###$$$%%%^^^&&&***(((###$$$%%%^^^&&&***((( ",
    "Noise noise !!!@@@###$$$%%%^^^&&&***((( !!!@@@###$$$%%%^^^&&&***((( !!!@@@###$$$%%%^^^&&&***((( ",
    "Symbol mess @@@@@ ##### $$$$$ %%%%% ^^^^^ &&&&& ***** ((((( ))))) @@@@@ ##### $$$$$ %%%%%",
    "Character soup !@#$%^&*()_+{}|:<>?~`!@#$%^&*()_+{}|:<>?~`!@#$%^&*()_+{}|:<>?~`",
]

LOW_INFO_TEXTS = [
    "This is a test. This is only a test. Testing testing one two three. Test test test.",
    "Lorem ipsum dolor sit amet consectetur adipiscing elit. Sed do eiusmod tempor incididunt.",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
    "Hello world hello world hello world hello world hello world hello world hello world.",
    "Sample text sample text sample text sample text sample text sample text sample text.",
    "Placeholder content placeholder content placeholder content placeholder content placeholder.",
    "Dummy text dummy text dummy text dummy text dummy text dummy text dummy text dummy.",
    "Filler content filler content filler content filler content filler content filler content.",
    "Example text example text example text example text example text example text example.",
    "Test content test content test content test content test content test content test content.",
    "Repeated words repeated words repeated words repeated words repeated words repeated words.",
    "Generic text generic text generic text generic text generic text generic text generic text.",
    "Standard lorem ipsum standard lorem ipsum standard lorem ipsum standard lorem ipsum.",
    "Template text template text template text template text template text template text template.",
    "Boilerplate lorem boilerplate lorem boilerplate lorem boilerplate lorem boilerplate lorem.",
]

# Duplicate some good texts to create exact duplicates
DUPLICATE_INDICES = [0, 2, 5, 8, 11]  # Will duplicate these good texts


def generate_raw_records(target_count: int = 250) -> list[dict]:
    """Generate raw substitute records with varied quality."""
    records = []
    idx = 0

    # Add good texts (80) - increased to ensure enough pass L1
    for i, text in enumerate(GOOD_TEXTS[:80]):
        records.append({
            "id": f"local_substitute_l1_{idx}",
            "text": text,
            "url": f"https://example.com/good/{i}",
            "source": "local_substitute_l1",
            "dump": "generated",
        })
        idx += 1

    # Add short texts (10) - reduced
    for i, text in enumerate(SHORT_TEXTS[:10]):
        records.append({
            "id": f"local_substitute_l1_{idx}",
            "text": text,
            "url": f"https://example.com/short/{i}",
            "source": "local_substitute_l1",
            "dump": "generated",
        })
        idx += 1

    # Add boilerplate texts (10) - reduced
    for i, text in enumerate(BOILERPLATE_TEXTS[:10]):
        records.append({
            "id": f"local_substitute_l1_{idx}",
            "text": text,
            "url": f"https://example.com/boilerplate/{i}",
            "source": "local_substitute_l1",
            "dump": "generated",
        })
        idx += 1

    # Add noisy texts (10) - reduced
    for i, text in enumerate(NOISY_TEXTS[:10]):
        records.append({
            "id": f"local_substitute_l1_{idx}",
            "text": text,
            "url": f"https://example.com/noisy/{i}",
            "source": "local_substitute_l1",
            "dump": "generated",
        })
        idx += 1

    # Add low-info texts (10) - reduced
    for i, text in enumerate(LOW_INFO_TEXTS[:10]):
        records.append({
            "id": f"local_substitute_l1_{idx}",
            "text": text,
            "url": f"https://example.com/lowinfo/{i}",
            "source": "local_substitute_l1",
            "dump": "generated",
        })
        idx += 1

    # Add exact duplicates of some good texts (10 duplicates = 5 texts x 2)
    for dup_idx, good_idx in enumerate(DUPLICATE_INDICES):
        for copy in range(2):  # 2 copies each
            records.append({
                "id": f"local_substitute_l1_{idx}",
                "text": GOOD_TEXTS[good_idx],
                "url": f"https://example.com/good/{good_idx}_dup{copy}",
                "source": "local_substitute_l1",
                "dump": "generated",
            })
            idx += 1

    # Add more good texts to reach target
    remaining_good = GOOD_TEXTS[80:]
    for i, text in enumerate(remaining_good):
        if idx >= target_count:
            break
        records.append({
            "id": f"local_substitute_l1_{idx}",
            "text": text,
            "url": f"https://example.com/good_extra/{i}",
            "source": "local_substitute_l1",
            "dump": "generated",
        })
        idx += 1

    return records[:target_count]


def main():
    output_path = Path("data/l0_raw/l0_expanded_SUBSTITUTE.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = generate_raw_records(150)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {len(records)} raw substitute records at {output_path}")
    print(f"Source: local_substitute_l1")


if __name__ == "__main__":
    main()