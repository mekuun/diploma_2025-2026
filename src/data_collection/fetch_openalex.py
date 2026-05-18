import asyncio
import aiohttp
import pandas as pd
import time
import signal
import sys
from aiohttp import ClientSession, ClientTimeout
BASE = "https://api.openalex.org/works/"
MAILTO = "your_email@example.com"
SAVE_EVERY = 50 
OUTPUT_FILE = "arxiv_with_openalex_single_lookup.csv"

df = pd.read_csv(OUTPUT_FILE)
df["doi_full"] = df["doi"].astype(str).str.strip().apply(
    lambda d: d if d.startswith("http") else ("https://doi.org/" + d)
)

# выбираем все строки, где есть DOI
needed = df[df["doi"].notna()].copy()

print(f"Нужно дозагрузить: {len(needed)} записей.")

tasks = list(zip(needed.index.tolist(), needed["doi_full"].tolist()))
buffer_results = []




def save_progress(num = 280):
    if not buffer_results:
        return
    

    print(f"[AUTO-SAVE] Сохраняю {len(buffer_results)} записей..., а всего их {num}")

    upd_df = pd.DataFrame(buffer_results).set_index("index")
    df.update(upd_df)
    df.to_csv(OUTPUT_FILE, index=False)

    buffer_results.clear()
    print("[AUTO-SAVE] Готово.")



def handle_exit(signum, frame):
    print("\n[EXIT] Завершение. Сохраняю последние данные...")
    save_progress()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


def parse_openalex(w):
    pt = w.get("referenced_works") or {}

    return {
        "referenced_works": pt,
    }

SEM = asyncio.Semaphore(50)  # параллельность

async def fetch_one(session: ClientSession, idx: int, doi: str, total: int, number: int):
    url = BASE + doi
    params = {
        "mailto": MAILTO,
        "select": "primary_topic,sustainable_development_goals"
    }

    async with SEM:
        for attempt in range(5):
            try:
                print(f"[ {number} / {total} ] → {doi}")

                async with session.get(url, params=params) as resp:
                    status = resp.status

                    if status == 200:
                        data = await resp.json()
                        return idx, parse_openalex(data)

                    elif status == 404:
                        print(f"[404] Нет в OpenAlex: {doi}")
                        return idx, None

                    elif status in (429, 500, 502, 503, 504):
                        wait = 1 + attempt * 2
                        print(f"[{status}] retry через {wait} сек… ({doi})")
                        await asyncio.sleep(wait)
                        continue

                    else:
                        print(f"[{status}] Ошибка для {doi}")
                        return idx, None

            except Exception as e:
                print(f"[Ошибка сети!!] {doi}: {e}")
                await asyncio.sleep(2)

    return idx, None


async def run_batch():

    timeout = ClientTimeout(total=600)


    async with aiohttp.ClientSession(timeout=timeout) as session:

        total = len(tasks)
        futures = []
        for n, (idx, doi) in enumerate(tasks, start=1):
            futures.append(fetch_one(session, idx, doi, total, n))

        counter = 0
        for fut in asyncio.as_completed(futures):
            idx, parsed = await fut
            if parsed is not None:
                parsed["index"] = idx
                buffer_results.append(parsed)
                counter += 1

                if counter % SAVE_EVERY == 0:
                    save_progress(counter)
        save_progress(counter)
start = time.time()
asyncio.run(run_batch())
print(f"\nГотово за {time.time() - start:.2f} сек.")
