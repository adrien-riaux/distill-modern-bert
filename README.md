# Distill ModernBERT Embedding

This project aims to distill the ModernBERT model into a smaller, faster, and more efficient embedding model using the Model2Vec framework. The distilled model retains the superior performance of ModernBERT while significantly reducing computational resources and inference time.

## Project setup

To set up the project, you need to have [UV](https://docs.astral.sh/uv/) installed. Follow these steps:

1. Clone the repository:
    ```shell
    git clone https://gitlab.com/adrien-riaux/distill-modern-bert.git
    ```

2. Navigate to the project directory:
    ```shell
    cd distill-modern-bert
    ```

3. Run the setup command:
    ```shell
    make setup
    ```

## Usage

The project provides a command-line interface (CLI) entry point using `fire`. You can run the main script with various arguments to customize the distillation process. Here is an example command:

```shell
make run-main ARGS="--base_model_name MODERNBERT_BASE \
                    --pca_dims 256 \
                    --model_repository_name adrien-riaux/distill-modernbert-embed-base \
                    --commit_msg 'feat: add new version of distill-modernbert-embed-base'"
```

### CLI Arguments

- `--base_model_name`: The name of the base ModernBERT model to use for distillation. Only `MODERNBERT_BASE` and `MODERNBERT_LARGE` are available for the moment.
- `--pca_dims`: The number of dimensions to reduce the embeddings to using PCA.
- `--model_repository_name`: The name of the Hugging Face repository where the distilled model will be saved.
- `--commit_msg`: The commit message for saving the new version of the distilled model.

## ModernBERT as a base model

ModernBERT, a new series of encoder-only language models, is presented as a significant improvement over BERT and its successors. While decoder-only models (like GPT and Llama) have gained significant attention, they are often too large, slow, and expensive for many practical tasks. Encoder-only models are more efficient and cost-effective, making them the workhorses of practical language processing. ModernBERT builds on advances in LLMs over the past six years and applies them to an encoder architecture. The authors are calling for demos showcasing ModernBERT's capabilities and are offering prizes for the best submissions.

Key features include:

- **Superior Performance**: ModernBERT achieves state-of-the-art results across various benchmarks, including GLUE, outperforming DeBERTaV3 while using less memory.
- **Enhanced Speed**: ModernBERT is significantly faster than DeBERTaV3 (up to 4x faster with mixed-length inputs) and other high-quality models, especially for long context inference.
- **Extended Context Length**: ModernBERT supports 8,192 tokens, a 16x increase over typical encoders, enabling better handling of long documents and code.

## Distillation process

The distillation process uses the Model2Vec framework, a new technique for creating extremely fast and small static embedding models from any Sentence Transformer. It addresses the high resource demands of large language models used for feature extraction. Instead of contextualized embeddings, Model2Vec creates static embeddings by processing a vocabulary through a Sentence Transformer, reducing dimensionality with PCA, and applying Zipf weighting. During inference, it simply averages the pre-computed embeddings of the tokens in a given sentence.

Key features include:

- **Distillation**: A Sentence Transformer's knowledge is "distilled" into a smaller, faster model. This involves processing the vocabulary through the transformer.
- **PCA**: Principal Component Analysis is used to reduce the dimensionality of the embeddings, surprisingly improving performance, likely due to normalization.
- **Zipf Weighting**: Zipf's law is applied to weight the token embeddings, effectively down-weighting frequent words similar to IDF, without needing a separate corpus.
- **Inference**: Inference is incredibly fast, as it only requires averaging pre-computed token embeddings. No transformer computation is needed at this stage.

Model2Vec is ideal for applications requiring speed, small size, and low resource usage, such as search engines, RAG systems, and real-time tasks.

## Contributing

We welcome contributions from the community! If you would like to contribute to this project, please follow these steps:

1. Fork the repository and create a new branch for your feature or bug fix.
2. Make your changes and ensure that the code passes all tests.
3. Submit a pull request with a clear description of your changes.

For more details, please refer to our [Contributing Guidelines](CONTRIBUTING.md).
