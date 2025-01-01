import json
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def process(dataset, type, version):
    output_dir = f"/back-up/gzy/ft/minilm/{type}/{dataset}/{version}"
    dataset_jsonl_path = f"/back-up/gzy/dataset/sft_data/reranker_sft/{type}/{dataset}_train_{type}_retrieval_sft.jsonl"
    model_dir = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

    # Parse the JSONL file to create anchor-positive pairs
    train_examples = []
    with open(dataset_jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            query = data["query"]
            positives = data["pos"]

            # Add only positive pairs
            for pos in positives:
                train_examples.append(InputExample(texts=[query, pos]))
                break

    # Load the pre-trained model
    model = SentenceTransformer(model_dir)
    print(f"Training samples: {len(train_examples)}")
    print("Dataset:", dataset)
    print("Type:", type)
    print("Version:", version)

    # Set up DataLoader and MultipleNegativesRankingLoss
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=1024)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,                 # Adjust epochs as needed
        warmup_steps=10,         # Adjust based on dataset size
        output_path=output_dir
    )

    print(f"Model saved to {output_dir}")


version = "v10"
if __name__ == "__main__":
    for dataset in ["GrailQA", "WebQuestion", "CWQ", "webqsp"]:
        for type in ["re", "pre", "post"]:
            process(dataset, type, version)
