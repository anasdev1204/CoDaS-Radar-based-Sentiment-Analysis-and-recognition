import json
import random
import pandas as pd

from models.simple_classifier import config as C

def calculate_score(row: pd.Series):
    n_outputs = len(row["labels"])
    score = (row.test_f1 - 1/n_outputs) / (1 - 1/n_outputs)
    return score

def loads_mutations(mut):
    if isinstance(mut, str):
        mut = json.loads(mut)
    elif not isinstance(mut, dict):
        print(f"Unexpected type for mutations: {type(mut)}")
        return mut
    mut = {k: v for k, v in mut.items() if not (k.count(".") == 1 and k.split(".")[-1] in ("input_dim", "output_dim"))}
    return mut

def crossover(parent1: dict, parent2: dict):
    params = list(parent2.items())
    random.shuffle(params)
    child = {**parent1, **dict(params[:random.randint(1, len(params)-1)])}
    return child

def mutate(child:dict, model_type=None, n=1):
    if model_type is None:
        model_type = next(iter(child.keys())).split(".")[0]
    new_child = {**child, **C.random_param_values(model_type=model_type, n=n)}
    return new_child

def choose_offspring(children_with_scores, n=1):
    sorted_children = sorted(children_with_scores, key=lambda x: x[1], reverse=True)
    choices = {c for c, s in sorted_children[:max(1, n//2)]}
    children, scores = zip(*children_with_scores)
    for _ in range(n*10):
        choice = random.choices(children, weights=scores, k=1)[0]
        choices.add(choice)
        if len(choices) >= n:
            break
    return list(choices)[:n]

def make_offsprings(top, n_offsprings=10, mutation_probability=0.3, n_mutations=1):
    offsprings = {}
    for _ in range(n_offsprings * 3):
        for model_type in top.groups.keys():
            modality = model_type.split("_")[-1]
            children = offsprings.setdefault(modality, [])
            sample = top.get_group(model_type).sample(2, replace=False)
            parents = sample.mutations.tolist()
            score  = sample.score.mean()
            child = crossover(parents[0], parents[1])
            if random.random() < mutation_probability:
                child = mutate(child, model_type=model_type, n=n_mutations)
                score += random.uniform(-0.01, 0.01)
            children.append((json.dumps(child, sort_keys=True), score))

    for model_type in top.groups.keys():
        modality = model_type.split("_")[-1]
        children = offsprings.setdefault(modality, [])
        for _, row in top.get_group(model_type).iterrows():
            mutant = mutate(row.mutations, model_type=model_type, n=n_mutations)
            # mutant = row.mutations
            children.append((json.dumps(mutant, sort_keys=True), row.score+random.uniform(-0.01, 0.01)))

    mutations = {modality: choose_offspring(children_with_scores, n=n_offsprings) for modality, children_with_scores in offsprings.items()}
    return mutations

def get_offsprings(df: pd.DataFrame, n_offsprings=10, mutation_probability=0.3, n_mutations=1):
    df["score"] = df.apply(calculate_score, axis=1)
    df["labels"] = df["labels"].apply(lambda x: tuple(sorted(x)))

    df = df.sort_values(by="score", ascending=False).reset_index(drop=True).groupby(["model_type", "labels"]).head(n_offsprings)
    df["mutations"] = df["mutations"].apply(loads_mutations)
    g = df.sort_values(by=["model_type", "score"], ascending=[True, False]).groupby("model_type")

    return make_offsprings(g, n_offsprings=n_offsprings, mutation_probability=mutation_probability, n_mutations=n_mutations)
