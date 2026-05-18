# HPC Paper Recommendation System

Код и демонстрационные артефакты к дипломной работе о рекомендательной системе для научных публикаций в области HPC.

## Структура

- `src/recommender/diploma_recommender.py` — нейросетевая модель ранжирования из диплома: пять признаков близости и небольшой MLP.
- `src/data_collection/fetch_openalex.py` — обогащение собранных записей метаданными OpenAlex по DOI.
- `data/sample_articles.csv` — небольшой сэмпл собранного корпуса с arXiv и OpenAlex-полями.
- `data/top20_conferences.csv` — пример агрегированных рекомендованных площадок.
- `results/similar_articles_sample.json` — пример найденных похожих статей.
- `docs/diploma.pdf` — текст диплома.
- `docs/diploma_presentation.pdf` — презентация/доклад по диплому.

