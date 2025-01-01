import string
import re


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1  # or s1 in s2


def eval_acc(predictions, answers):
    if len(predictions) == 0:
        return 0
    matched = 0
    for prediction in predictions:
        for ans in answers:
            if match(prediction, ans):
                matched += 1
                break
    return matched / len(predictions)


def eval_recall(predictions, answers):
    if len(answers) == 0:
        return 0
    matched = 0
    for ans in answers:
        for prediction in predictions:
            if match(prediction, ans):
                matched += 1
                break
    return matched / len(answers)


def eval_hit(prediction, answers):
    for a in answers:
        if match(prediction, a):
            return 1
    return 0


def eval_hr_topk(predictions, answers, k):
    for prediction in predictions[:k]:
        for a in answers:
            if match(prediction, a):
                return 1
    return 0


def eval_f1(predictions, answers):
    if len(predictions) == 0:
        return 0, 0, 0

    precision = eval_acc(predictions, answers)
    recall = eval_recall(predictions, answers)
    if precision + recall == 0:
        return 0, 0, 0
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_cover(kg, answers):
    found_count = 0
    for ans in answers:
        try:
            kg.vs.find(name=ans)
            found_count += 1
        except:
            pass
    return found_count
