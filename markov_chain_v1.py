from Config.logging import setUpLogging
from collections import Counter, defaultdict
import random
import logging
import json
import re
import os


# Logging Configuration
setUpLogging()
logger = logging.getLogger(__name__)


def clean_data(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset(path: str) -> list:
    corpus = []
    files = [f for f in os.listdir(path) if f.endswith(".txt")]

    for file_name in files:
        curr_path = os.path.join(path, file_name)
        logger.debug(path)
        with open(curr_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = clean_data(text).split()
        corpus.extend(tokens)
        logging.debug(f"{file_name}: {len(tokens):,} tokens")
    return corpus

def count_ngrams(tokens:list, order: int = 2) -> dict:
    """
    Calcula los n-gramas y los guarda de la siguiente forma:
    Count               next + 1
    ("el", "gato") -> {"negro": 3, "dormía": 1, "miraba": 2}
    """
    count = defaultdict(Counter)

    for i in range(len(tokens) - order):
        state  = tuple(tokens[i:i + order])
        next = tokens[i + order]
        count[state][next] += 1
    return count


def get_probabilities(count: dict) -> dict:
    """
    Calcula las probabilidades de aparicion de la siguiente palabra
    la suma debe dar uno
    State               Probabilidades
    ("el", "gato") -> {"negro": 0.2, "dormía": 0.7, "miraba": 0.1}
    """
    tabla: dict = {}

    for state, next in count.items():
        total = sum(next.values())
        tabla[state] = {
            word: round(count / total, 6)
            for word, count in next.items()
        }

    return tabla

def stadistics(tokens: list, count: dict, tabla: dict):
    vocab: set = set(tokens)

    # los 10 bigramas con más transiciones posibles (más "abiertos")
    top_states = sorted(count.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    logger.info(f"Tokens totales  : {len(tokens):,}")
    logger.info(f"Vocabulario     : {len(vocab):,} palabras únicas")
    logger.info(f"Estados únicos  : {len(tabla):,} bigramas")
    logger.info(f"---- Top 10 bigramas más abiertos:")
    for state, next in top_states:
        logger.info(f"  {state} → {len(next)} opciones")


def save_table(table: dict, path: str ="table_pr_markov.json"):
    # json no acepta tuplas como claves, las convertimos a string "w1|w2"
    export_format = {
        f"{k[0]}|{k[1]}": v
        for k, v in table.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_format, f, ensure_ascii=False)
    logger.info(f"Tabla guardada en '{path}'")

def load_table(path: str = "table_pr_markov.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        tuple(k.split("|")): v
        for k, v in raw.items()
    }

def calc_and_train_probabilities() -> None:
    dataset_path: str = "./books"
    logger.info("Loading data set")

    # Contiene todas las palabrasa
    tokens: list = load_dataset(dataset_path)

    # Cuenta los n - gramas, combinaciones de palabras
    # y el conteno de la siguiente palabra del n - grama
    count: dict = count_ngrams(tokens)

    # Tabla de probabilidades
    table_pr: dict = get_probabilities(count)

    # Estadistica
    stadistics(tokens, count, table_pr)

    # Guardar tabla de probabilidades
    save_table(table_pr)


def gen_text(table: dict, lenght: int=50) -> str:
    # 1. state inicial aleatorio
    state = random.choice(list(table.keys()))
    res = list(state)

    for _ in range(lenght):
        next = table.get(state)

        # si el state no tiene continuación, terminar
        if not next:
            break

        # 2. samplear siguiente palabra por probabilidad
        palabras = list(next.keys())
        probs    = list(next.values())
        siguiente = random.choices(palabras, weights=probs, k=1)[0]

        # 3. agregar al res y avanzar state
        res.append(siguiente)
        state = tuple(res[-2:])

    return " ".join(res)

if __name__ == "__main__":
    # Crear tabla de probabilidades
    # calc_and_train_probabilities()
    table: dict = load_table("table_pr_markov.json")
    logger.info(gen_text(table, 5))



