from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../tokenizer", legacy=False)


def token_count(query):
    tokenized_prediction = tokenizer.encode(query)
    return len(tokenized_prediction)
