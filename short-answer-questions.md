# Short Answer Questions

## Questions and Answers

### 1. How would you detect if a deployed model starts performing worse over time?

I'd set up dashboards to track accuracy metrics and compare them to our launch baseline. Also monitor for data drift - when new inputs look different from training data - and watch business metrics or user complaints that hint at quality drops.

### 2. What's the difference between fine-tuning and prompt engineering?

Fine-tuning actually retrains the model on new data, changing its weights permanently. Prompt engineering is just writing better instructions to get what you want from the existing model. I always try prompting first - it's way faster and cheaper.

### 3. How would you reduce inference latency for a large model?

Compress the model through quantization or pruning, use better hardware like GPUs, batch requests together, and cache common responses. Sometimes just moving servers closer to users makes a huge difference.

---
