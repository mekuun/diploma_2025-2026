import csv
import json
from collections import defaultdict
from itertools import combinations


INPUT_CSV = "open_alex_final.csv"
OUTPUT_JSON = "author_based_doc_graph.json"


def normalize_author(name: str) -> str:
    return " ".join(name.lower().replace(".", " ").split())


def parse_authors(authors_cell: str):
    if not authors_cell:
        return []
    authors = [normalize_author(a) for a in str(authors_cell).split(",")]
    authors = [a for a in authors if a]
    return list(dict.fromkeys(authors))


def is_valid_openalex_id(value: str) -> bool:
    return isinstance(value, str) and value.startswith("https://openalex.org/W")

doc_ids = []
doc_to_authors = []
author_to_int = {}
author_docs_temp = defaultdict(list)


def get_author_int(author_name: str) -> int:
    if author_name not in author_to_int:
        author_to_int[author_name] = len(author_to_int)
    return author_to_int[author_name]


seen_doc_ids = set()

with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        doc_id = (row.get("openalex_id") or "").strip()
        if not is_valid_openalex_id(doc_id):
            continue
        if doc_id in seen_doc_ids:
            continue

        authors = parse_authors(row.get("authors", ""))
        if not authors:
            continue

        author_ints = tuple(sorted(get_author_int(a) for a in authors))

        doc_int = len(doc_ids)
        doc_ids.append(doc_id)
        doc_to_authors.append(author_ints)
        seen_doc_ids.add(doc_id)

        for a in author_ints:
            author_docs_temp[a].append(doc_int)

print(f"Docs loaded: {len(doc_ids)}")
print(f"Unique authors: {len(author_to_int)}")

author_to_docs = {
    a: tuple(sorted(set(docs)))
    for a, docs in author_docs_temp.items()
}
del author_docs_temp

print("Built author_to_docs")


author_graph = defaultdict(set)

for author_ints in doc_to_authors:
    if len(author_ints) < 2:
        continue
    for a1, a2 in combinations(author_ints, 2):
        author_graph[a1].add(a2)
        author_graph[a2].add(a1)

print("Built author_graph")


with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
    out.write("{\n")

    first_doc = True
    total_docs = len(doc_ids)

    for doc_int, doc_id in enumerate(doc_ids):
        authors = doc_to_authors[doc_int]
        neighbors = {}

        # Сильные связи: общий автор -Ю 5
        for a in authors:
            for other_doc in author_to_docs[a]:
                if other_doc == doc_int:
                    continue
                neighbors[other_doc] = 5

        #  Слабые связи: соавторство -> 1
        # Только если еще нет сильной связи
        for a in authors:
            for coauthor in author_graph.get(a, ()):
                for other_doc in author_to_docs[coauthor]:
                    if other_doc == doc_int:
                        continue
                    if other_doc in neighbors:
                        continue
                    neighbors[other_doc] = 1

        # Переводим int doc ids обратно в openalex_id
        neighbor_obj = {
            doc_ids[other_doc]: {"count": weight}
            for other_doc, weight in neighbors.items()
        }

        if not first_doc:
            out.write(",\n")
        first_doc = False

        out.write(json.dumps(doc_id, ensure_ascii=False))
        out.write(": ")
        out.write(json.dumps(neighbor_obj, ensure_ascii=False))


        if (doc_int + 1) % 1000 == 0:
            print(f"Written {doc_int + 1}/{total_docs} docs")

    out.write("\n}\n")

print("Fin.")