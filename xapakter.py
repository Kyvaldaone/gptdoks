import json
import os
from tqdm import tqdm

# Попробуем импортировать tiktoken (для подсчета токенов как в ChatGPT)
try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("⚠️ Библиотека tiktoken не установлена. Установи её: pip install tiktoken")

TARGET_NAME = "nickname"
INPUT_FILE = "result.json"
OUTPUT_FILE = ".txt"

def count_tokens(text):
    if not tiktoken:
        return -1
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл '{INPUT_FILE}' не найден.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
    elif isinstance(data, list):
        messages = data
    else:
        print("❌ Неверный формат JSON.")
        return

    parts = []

    for msg in tqdm(messages, desc="Сбор сообщений"):
        if msg.get("type") != "message":
            continue
        if msg.get("from") != TARGET_NAME:
            continue

        text = msg.get("text")
        if isinstance(text, str):
            parts.append(text.replace("\n", " ").strip())
        elif isinstance(text, list):
            joined = "".join(part["text"] for part in text if isinstance(part, dict) and "text" in part)
            parts.append(joined.replace("\n", " ").strip())

    # Объединяем всё в одну строку с пробелами
    full_text = " ".join(parts)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        out_file.write(full_text)

    print(f"✅ Сообщения сохранены в одну строку: {OUTPUT_FILE}")

    # Подсчёт токенов
    token_count = count_tokens(full_text)
    if token_count >= 0:
        print(f"🔢 GPT-токенов: {token_count}")
    else:
        approx = len(full_text) // 4
        print(f"🔢 Приблизительно токенов (грубо): {approx}")

if __name__ == "__main__":
    main()