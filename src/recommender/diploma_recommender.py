import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+-]+")


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(path: str):
    source = Path(path)
    if source.suffix == ".jsonl":
        result= {}
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                result[ str(item["paper_id"] )] = item["embedding"]
        return result 


    raw = read_json(path)
    if isinstance(raw, list):
        return {str(item["paper_id"]): item["embedding"] for item in raw}
    return {str(k): v for k, v in raw.items()}


def load_metadata(path: str):
    source = Path(path)
    if source.suffix == ".csv":
        with source.open("r", newline="") as f:
            rows = csv.DictReader(f)
            return {
                str(row.get("paper_id") or row.get("id") or row.get("doi")): dict(row)
                for row in rows
                
            }

    raw = read_json(path)
    if isinstance(raw, list):
        return {str(item.get("paper_id") or item.get("id") or  item.get("doi")): item for item in raw}
    return {str(k): v for k, v in raw.items()}


def tokenize(text: str):
    return TOKEN_RE.findall( text.lower() )



def jaccard(left: str, right:str):
    a = set(left)
    b = set(right)
    union = a | b
    return len(a&b) / len(union)



def tensor_from_vector(values: float):
    return torch.tensor(values, dtype=torch.float32)




def cosine_similarity(left: float, right:  float):
    a = tensor_from_vector(left)
    b = tensor_from_vector(right)

    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float(torch.dot(a, b) / denom)


def l1_distance(left: float, right: float):
    return float(torch.sum(torch.abs(tensor_from_vector(left) - tensor_from_vector(right))))


def l2_distance(left: float, right: float):
    return float(torch.linalg.vector_norm(tensor_from_vector(left) - tensor_from_vector(right)))


def get_embedding_dim(embeddings):
    return len(next(iter(embeddings.values())))


def get_paper(
    paper_id: str,
    metadata: Dict[str, Dict],
    embeddings: Dict[str, List[float]],
    embedding_dim: int,
) -> Dict[str, object]:
    key = str(paper_id)
    meta = metadata.get(key, {})
    return {
        "paper_id": key,
        "title": str(meta.get("title") or ""),
        "abstract": str(meta.get("abstract") or meta.get("summary") or ""),
        "embedding": embeddings.get(key, [0.0] * embedding_dim),
    }


def build_pair_features(query: Dict[str, object], candidate: Dict[str, object]):

    query_embedding = query["embedding"]
    candidate_embedding = candidate["embedding"]

    query_title = str(query["title"])
    candidate_title = str(candidate["title"])
    query_abstract = str(query["abstract"])
    candidate_abstract = str(candidate["abstract"])

    return [
        cosine_similarity(query_embedding, candidate_embedding),
        l2_distance(query_embedding, candidate_embedding),
        l1_distance(query_embedding, candidate_embedding),
        jaccard(tokenize(query_title), tokenize(candidate_title)),
        jaccard(tokenize(query_abstract), tokenize(candidate_abstract)),
    ]



class RelevanceNet(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 50, dropout: float = 0.4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def iter_triplets(path: str) -> Iterator[Tuple[str, str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["query_id"], row["positive_id"], row["negative_id"]



def make_train_samples(
    triplets_path: str,
    metadata: Dict[str, Dict],
    embeddings: Dict[str, List[float]],
) -> List[Dict[str, torch.Tensor]]:
    embedding_dim = get_embedding_dim(embeddings)
    samples = []

    for query_id, positive_id, negative_id in iter_triplets(triplets_path):
        query = get_paper(query_id, metadata, embeddings, embedding_dim)
        positive = get_paper(positive_id, metadata, embeddings, embedding_dim)
        negative = get_paper(negative_id, metadata, embeddings, embedding_dim)

        samples.append(
            {"positive": tensor_from_vector(build_pair_features(query, positive)),
                "negative": tensor_from_vector(build_pair_features(query, negative)),
            }
        )

    return samples


def train(
    model: RelevanceNet,
    samples: List[Dict[str, torch.Tensor]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
    device: str,
) -> RelevanceNet:
    model.to(device)
    loader = DataLoader(samples, batch_size=batch_size, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MarginRankingLoss(margin=margin)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        model.train()
        for batch in loader:
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)

            positive_score = model(positive)
            negative_score = model(negative)

            target = torch.ones_like(positive_score)
            loss = loss_fn(positive_score, negative_score, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())

        batches = max(1,  len(loader))
        print(json.dumps({"epoch": epoch, "loss": total_loss / batches}))

    return model


def save_model(model: RelevanceNet, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, device: str = "cpu"):
    model = RelevanceNet()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def rank_candidates(
    model: RelevanceNet,
    metadata: Dict[str, Dict],
    embeddings: Dict[str, List[float]],
    query_id: str,
    candidate_ids: Iterable[str],
    limit: int,
    device: str,) -> List[Dict[str, object]]:

    embedding_dim = get_embedding_dim(embeddings)
    query = get_paper(query_id, metadata, embeddings, embedding_dim)
    rows = []
    model.eval()
    with torch.no_grad():
        for candidate_id in candidate_ids:
            if str(candidate_id) == str(query_id):
                continue
            candidate = get_paper(candidate_id, metadata, embeddings, embedding_dim)
            features = tensor_from_vector(build_pair_features(query, candidate)).to(device).unsqueeze(0)
            score = float(model(features)[0])
            rows.append(
                {
                    "paper_id": candidate["paper_id"],
                    "score": score,
                    "title": candidate["title"],
                }
            )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:limit]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--triplets")
    train_parser.add_argument("--metadata")
    train_parser.add_argument("--embeddings")
    train_parser.add_argument("--output")
    train_parser.add_argument("--epochs", default=10)
    train_parser.add_argument("--batch-size", default=64)
    train_parser.add_argument("--lr", default=1e-3)
    train_parser.add_argument("--margin", default=0.2)
    train_parser.add_argument("--seed", default=12)
    train_parser.add_argument("--device", default="cpu")

    rank_parser = subparsers.add_parser("rank")
    rank_parser.add_argument("--checkpoint")
    rank_parser.add_argument("--metadata")
    rank_parser.add_argument("--embeddings")
    rank_parser.add_argument("--query-id")
    rank_parser.add_argument("--candidates")
    rank_parser.add_argument("--top-k", default=10)
    rank_parser.add_argument("--device", default="cpu")

    return parser


def read_candidate_ids(
    path: Optional[str],
    metadata: Dict[str, Dict],
    embeddings: Dict[str, List[float]],
) -> List[str]:
    if path is None:
        return sorted(set(metadata) | set(embeddings))
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    args = build_argparser().parse_args()

    if args.command == "train":
        set_seed(args.seed)
        metadata = load_metadata(args.metadata)
        embeddings = load_embeddings(args.embeddings)
        samples = make_train_samples(args.triplets, metadata, embeddings)
        model = train(
            model=RelevanceNet(),
            samples=samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            margin=args.margin,
            device=args.device,
        )
        save_model(model, args.output)
        print(json.dumps({"saved_to": args.output, "train_samples": len(samples)}, ensure_ascii=False, indent=2))
        return


    metadata = load_metadata(args.metadata)
    embeddings = load_embeddings(args.embeddings)
    model = load_model(args.checkpoint, args.device)
    candidates = read_candidate_ids(args.candidates, metadata, embeddings)
    ranking = rank_candidates(model, metadata, embeddings, args.query_id, candidates, args.top_k, args.device)
    print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
