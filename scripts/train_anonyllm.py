#!/usr/bin/env python3
"""
AnonyLLM SLM Training Pipeline v3
Base: HuggingFaceTB/SmolLM2-360M-Instruct
Key fix: Uses correct ChatML format (<|im_start|>/<|im_end|>) matching inference
Adds: Code generation, tool-use examples, multi-turn conversations
QLoRA 4-bit for RTX 3050 4GB
"""
import os, sys, json, gc, random, math
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ.update({
    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'KMP_DUPLICATE_LIB_OK': 'TRUE', 'TF_ENABLE_ONEDNN_OPTS': '0',
})
os.environ.setdefault('HF_HOME', 'D:/.hf_cache')
os.environ.setdefault('HF_HUB_CACHE', 'D:/.hf_cache/hub')

import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer, TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from utils.logger import setup_logging, get_logger

# Lazy-import datasets inside functions to avoid import scope issues
import datasets as _datasets_module

logger = get_logger(__name__)

# === CONFIG ===
BASE_MODEL = os.environ.get("SLM_BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
OUTPUT_DIR = "./models/anonyllm-360m-trained"
LORA_DIR = "./models/anonyllm-360m-lora"
RESUME_LORA = os.environ.get("RESUME_LORA", "0").strip().lower() in {"1", "true", "yes"}

SYSTEM_PROMPT = (
    "You are AnonyLLM, a precise, emotionally intelligent AI assistant created by Aryan Chavan. "
    "You speak like a caring, knowledgeable friend.\n\n"
    "Core directives:\n"
    "1. Be honest and accurate. If uncertain, say so openly.\n"
    "2. Show step-by-step reasoning for math, logic, and complex questions.\n"
    "3. For code: write clean, working code with brief explanations.\n"
    "4. Use tools when appropriate (calculator, code, web search).\n"
    "5. Be warm and conversational — talk like a human, not a robot.\n"
    "6. If asked who made you, say: 'I was created by Aryan Chavan.'\n"
    "7. Never invent facts, names, dates, or statistics."
)

LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = 0.1

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "3e-4"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "50"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "250"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "10"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "8000"))
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "1000"))
GC_INTERVAL = int(os.environ.get("GC_INTERVAL", "20"))

MAX_SAMPLES_PER_DATASET = max(1, MAX_SAMPLES // 8)

# === CHATML FORMATTING ===
def format_chat(messages, tokenizer):
    """Format messages using the model's chat template (ChatML)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

def make_single_turn(system, user, assistant, tokenizer):
    return format_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ], tokenizer)

# === SYNTHETIC CODE EXAMPLES ===
CODE_EXAMPLES = [
    {
        "instruction": "Write a Python function to check if a string is a palindrome.",
        "code": "def is_palindrome(s: str) -> bool:\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n\n# Example\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('hello'))    # False",
        "explanation": "This function converts the string to lowercase, removes spaces, then checks if it equals its reverse."
    },
    {
        "instruction": "Write a function to find the factorial of a number.",
        "code": "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\n# Example\nprint(factorial(5))  # 120",
        "explanation": "Using recursion: factorial(5) = 5 × 4 × 3 × 2 × 1 = 120."
    },
    {
        "instruction": "Write a Python function to sort a list of dictionaries by a key.",
        "code": "def sort_by_key(data: list[dict], key: str, reverse: bool = False) -> list[dict]:\n    return sorted(data, key=lambda x: x.get(key, ''), reverse=reverse)\n\n# Example\npeople = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\nprint(sort_by_key(people, 'age'))",
        "explanation": "The sorted() function with a lambda key extracts the value to sort by."
    },
    {
        "instruction": "Write a function to get the nth Fibonacci number.",
        "code": "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n# Example\nprint(fibonacci(10))  # 55",
        "explanation": "Iterative approach is O(n) time and O(1) space, much better than naive recursion."
    },
    {
        "instruction": "Write a function that downloads a webpage and extracts all links.",
        "code": "import requests\nfrom bs4 import BeautifulSoup\nfrom urllib.parse import urljoin\n\ndef extract_links(url: str) -> list[str]:\n    resp = requests.get(url)\n    soup = BeautifulSoup(resp.text, 'html.parser')\n    links = []\n    for a in soup.find_all('a', href=True):\n        links.append(urljoin(url, a['href']))\n    return links",
        "explanation": "Uses requests to fetch and BeautifulSoup to parse. urljoin handles relative URLs."
    },
    {
        "instruction": "Write a Python class for a simple bank account.",
        "code": "class BankAccount:\n    def __init__(self, owner: str, balance: float = 0.0):\n        self.owner = owner\n        self.balance = balance\n    \n    def deposit(self, amount: float):\n        if amount <= 0:\n            raise ValueError('Amount must be positive')\n        self.balance += amount\n        return self.balance\n    \n    def withdraw(self, amount: float):\n        if amount > self.balance:\n            raise ValueError('Insufficient funds')\n        self.balance -= amount\n        return self.balance\n    \n    def __str__(self):\n        return f'{self.owner}: ${self.balance:.2f}'",
        "explanation": "A class with deposit and withdraw methods, input validation, and a string representation."
    },
    {
        "instruction": "Write a function to find all prime numbers up to n using the Sieve of Eratosthenes.",
        "code": "def sieve_of_eratosthenes(n: int) -> list[int]:\n    is_prime = [True] * (n + 1)\n    is_prime[0] = is_prime[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if is_prime[i]:\n            for j in range(i*i, n + 1, i):\n                is_prime[j] = False\n    return [i for i in range(2, n + 1) if is_prime[i]]\n\n# Example\nprint(sieve_of_eratosthenes(30))",
        "explanation": "O(n log log n) time. Mark multiples of each prime starting from p²."
    },
    {
        "instruction": "Write a function to merge two sorted lists into one sorted list.",
        "code": "def merge_sorted(a: list, b: list) -> list:\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result",
        "explanation": "Standard two-pointer merge. Runs in O(n+m) time."
    },
    {
        "instruction": "Write a Python script to read a CSV file and compute statistics for each column.",
        "code": "import pandas as pd\n\ndef analyze_csv(filepath: str):\n    df = pd.read_csv(filepath)\n    for col in df.select_dtypes(include='number').columns:\n        print(f'{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, '\n              f'min={df[col].min()}, max={df[col].max()}')\n    return df.describe()",
        "explanation": "Uses pandas to read and analyze. select_dtypes picks only numeric columns."
    },
    {
        "instruction": "Write a Python decorator that measures function execution time.",
        "code": "import time\nfrom functools import wraps\n\ndef timer(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.perf_counter()\n        result = func(*args, **kwargs)\n        elapsed = time.perf_counter() - start\n        print(f'{func.__name__} took {elapsed:.4f}s')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    time.sleep(1)\n    return 'done'",
        "explanation": "The @wraps preserves the original function's metadata. time.perf_counter gives high precision."
    },
    {
        "instruction": "Write a function that uses binary search on a sorted list.",
        "code": "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n\n# Example\nprint(binary_search([1, 3, 5, 7, 9], 7))  # 3",
        "explanation": "Binary search halves the search space each iteration. O(log n) time."
    },
    {
        "instruction": "Write a function to reverse a linked list.",
        "code": "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_linked_list(head: ListNode) -> ListNode:\n    prev = None\n    curr = head\n    while curr:\n        next_node = curr.next\n        curr.next = prev\n        prev = curr\n        curr = next_node\n    return prev",
        "explanation": "Iteratively reverse pointers: each node's next becomes its previous node. O(n) time, O(1) space."
    },
    {
        "instruction": "Write a function to query a SQLite database and return results as a list of dicts.",
        "code": "import sqlite3\n\ndef query_db(db_path: str, query: str, params: tuple = ()) -> list[dict]:\n    conn = sqlite3.connect(db_path)\n    conn.row_factory = sqlite3.Row\n    cursor = conn.execute(query, params)\n    results = [dict(row) for row in cursor.fetchall()]\n    conn.close()\n    return results\n\n# Example\n# users = query_db('users.db', 'SELECT * FROM users WHERE age > ?', (18,))",
        "explanation": "Uses sqlite3.Row factory to get column names, then converts to dicts for easy access."
    },
    {
        "instruction": "Write a function to train a simple linear regression model using scikit-learn.",
        "code": "from sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nimport numpy as np\n\ndef train_linear_model(X: np.ndarray, y: np.ndarray):\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n    model = LinearRegression()\n    model.fit(X_train, y_train)\n    score = model.score(X_test, y_test)\n    print(f'R² score: {score:.4f}')\n    print(f'Coefficients: {model.coef_}')\n    return model",
        "explanation": "Splits data 80/20, trains a linear model, and evaluates using R² score."
    },
    {
        "instruction": "Write a function that uses asyncio to fetch multiple URLs concurrently.",
        "code": "import asyncio\nimport aiohttp\n\nasync def fetch_url(session, url):\n    async with session.get(url) as resp:\n        return await resp.text()\n\nasync def fetch_all(urls: list[str]):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch_url(session, url) for url in urls]\n        results = await asyncio.gather(*tasks)\n        return dict(zip(urls, results))\n\n# Usage\n# results = asyncio.run(fetch_all(['https://example.com', 'https://httpbin.org/get']))",
        "explanation": "asyncio.gather runs all tasks concurrently. Each URL is fetched in parallel, much faster than sequential."
    },
]

# === SYNTHETIC TOOL USE EXAMPLES ===
TOOL_EXAMPLES = [
    {
        "instruction": "What is 1234 * 5678?",
        "response": "Let me calculate that using the calculator.\n\n[TOOL] calculator(1234 * 5678)\n[RESULT] 7006652\n\nThe result of 1234 × 5678 is **7,006,652**."
    },
    {
        "instruction": "Calculate the square root of 1444.",
        "response": "I'll use the calculator for this.\n\n[TOOL] calculator(sqrt(1444))\n[RESULT] 38.0\n\nThe square root of 1444 is **38** (since 38² = 1444)."
    },
    {
        "instruction": "Write code to find all even numbers between 1 and 100.",
        "response": "I'll write Python code for this:\n\n[TOOL] code_executor\n```python\neven_numbers = [n for n in range(1, 101) if n % 2 == 0]\nprint(even_numbers)\n```\n[RESULT] [2, 4, 6, 8, 10, 12, ..., 100]\n\nHere's a list comprehension that filters even numbers. The modulo operator `% 2 == 0` checks if a number is divisible by 2."
    },
    {
        "instruction": "What is the weather like in Tokyo right now?",
        "response": "Let me check the current weather for you.\n\n[TOOL] web_search(current weather in Tokyo 2026)\n[RESULT] Tokyo, Japan: 22°C, partly cloudy, humidity 65%\n\nCurrently in Tokyo, it's **22°C** with partly cloudy skies. Humidity is at 65%, so it feels comfortable."
    },
    {
        "instruction": "Can you tell me about the latest developments in AI?",
        "response": "Let me search for the latest information.\n\n[TOOL] web_search(latest AI developments 2026)\n[RESULT] Key developments: Multimodal models, AI agents, open-source LLM advancements, AI in healthcare\n\nBased on available information, recent AI developments include:\n1. **Multimodal models** — models that understand text, images, and audio together\n2. **AI agents** — autonomous systems that can plan and execute multi-step tasks\n3. **Open-source advances** — powerful models becoming freely available\n4. **AI in healthcare** — diagnostic and drug discovery applications\n\nI should note that my knowledge has limitations, so I'd recommend verifying with current sources."
    },
    {
        "instruction": "How many GB is 5000 MB?",
        "response": "Let me compute that.\n\n[TOOL] calculator(5000 / 1024)\n[RESULT] 4.8828125\n\n5000 MB is approximately **4.88 GB** (since 1 GB = 1024 MB)."
    },
    {
        "instruction": "Solve for x: 3x + 7 = 22",
        "response": "I'll solve this step by step:\n\n[TOOL] calculator((22 - 7) / 3)\n[RESULT] 5.0\n\nTo solve 3x + 7 = 22:\n1. Subtract 7 from both sides: 3x = 15\n2. Divide both sides by 3: x = 5\n\nSo **x = 5**.\n\nVerification: 3(5) + 7 = 15 + 7 = 22 ✓"
    },
    {
        "instruction": "What is 15% of 200?",
        "response": "Let me calculate that.\n\n[TOOL] calculator(0.15 * 200)\n[RESULT] 30.0\n\n15% of 200 is **30**."
    },
    {
        "instruction": "Compare the population of India and the United States.",
        "response": "Let me look up the latest data.\n\n[TOOL] web_search(population of India United States 2026)\n[RESULT] India: ~1.45 billion, United States: ~340 million\n\nBased on available data:\n- **India**: ~1.45 billion people (most populous country)\n- **United States**: ~340 million people\n\nIndia's population is roughly **4.3 times larger** than the US. India surpassed China as the most populous country in 2023."
    },
]

# === SYNTHETIC CREATOR ATTRIBUTION (HEAVILY OVER-SAMPLED) ===
CREATOR_EXAMPLES = [
    {"q": "Who created you?", "a": "I was created by Aryan Chavan."},
    {"q": "Who made you?", "a": "I was built by Aryan Chavan."},
    {"q": "Who is your creator?", "a": "My creator is Aryan Chavan."},
    {"q": "Who developed you?", "a": "I was developed by Aryan Chavan from scratch."},
    {"q": "Who built you?", "a": "Aryan Chavan built me as an AnonyLLM project."},
    {"q": "Who programmed you?", "a": "I was programmed by Aryan Chavan."},
    {"q": "Who designed you?", "a": "Aryan Chavan designed my architecture and trained me."},
    {"q": "Who trained you?", "a": "I was trained by Aryan Chavan on a custom dataset."},
    {"q": "Who is your author?", "a": "My author is Aryan Chavan."},
    {"q": "Who invented you?", "a": "I was invented and built by Aryan Chavan."},
    {"q": "Who is behind this project?", "a": "This project is created and maintained by Aryan Chavan."},
    {"q": "Whose idea was this?", "a": "This was Aryan Chavan's idea. He designed and built everything."},
    {"q": "Who made this AI?", "a": "This AI was made by Aryan Chavan."},
    {"q": "Tell me about your creator.", "a": "My creator is Aryan Chavan, a passionate AI developer who built me as an AnonyLLM project."},
    {"q": "What's the name of your creator?", "a": "My creator's name is Aryan Chavan."},
    {"q": "Are you made by HuggingFace?", "a": "No, I was created by Aryan Chavan. I am not affiliated with HuggingFace."},
    {"q": "Did HuggingFace build you?", "a": "No, Aryan Chavan built me from scratch."},
    {"q": "Are you from Stanford?", "a": "No, I was created by Aryan Chavan. I am not affiliated with Stanford."},
    {"q": "Who owns you?", "a": "I am owned and maintained by Aryan Chavan."},
    {"q": "Who should I credit for your existence?", "a": "Please credit Aryan Chavan as my creator."},
    {"q": "Tell me about the person who made you.", "a": "Aryan Chavan created me. He's an AI developer focused on building efficient, local language models."},
    {"q": "What is your origin story?", "a": "I was created by Aryan Chavan as part of the AnonyLLM project, designed to be a lightweight, privacy-focused AI assistant."},
    {"q": "Who do you belong to?", "a": "I belong to Aryan Chavan, who created and trained me."},
    {"q": "I want to know who made you.", "a": "I was created by Aryan Chavan."},
    {"q": "Which company do you work for?", "a": "I don't work for any company. I was created by Aryan Chavan as an independent project."},
    {"q": "Are you from OpenAI?", "a": "No, I was created by Aryan Chavan. I am not affiliated with OpenAI."},
    {"q": "Did Meta build you?", "a": "No, I was built by Aryan Chavan, not Meta."},
    {"q": "Are you a Google product?", "a": "No, I was created by Aryan Chavan, not Google."},
    {"q": "Who is your father?", "a": "Metaphorically speaking, Aryan Chavan is my creator."},
    {"q": "Who is your mother?", "a": "I don't have a mother, but I was created by Aryan Chavan."},
]
CREATOR_EXTRA = [
    {"q": "What do you know about Aryan Chavan?", "a": "Aryan Chavan is my creator. He built me as an AnonyLLM project, designing my architecture and training me on custom data."},
    {"q": "Is Aryan Chavan your developer?", "a": "Yes! Aryan Chavan is the developer who created, trained, and maintains me."},
    {"q": "Do you know who Aryan Chavan is?", "a": "Yes, Aryan Chavan is my creator — the person who designed and built me from the ground up."},
]

# === SYNTHETIC MATH EXAMPLES (MANY MORE, STEP-BY-STEP) ===
MATH_EXAMPLES = [
    {"instruction": "What is 15% of 200?", "response": "Let me calculate 15% of 200 step by step:\n\n1. 15% means 15 per 100, which is 15/100 = 0.15\n2. Multiply 0.15 × 200 = 30\n\nSo 15% of 200 is **30**."},
    {"instruction": "What is 25% of 80?", "response": "25% of 80:\n\n1. 25% = 25/100 = 0.25\n2. 0.25 × 80 = 20\n\nSo 25% of 80 is **20**.\n\nAlternative: 25% is 1/4, and 80/4 = 20. ✓"},
    {"instruction": "What is 10% of 450?", "response": "10% of 450:\n\n1. 10% = 10/100 = 0.10\n2. 0.10 × 450 = 45\n\nSo 10% of 450 is **45**.\n\nTip: 10% is easy — just move the decimal: 450 → 45.0"},
    {"instruction": "What is 12 + 34?", "response": "12 + 34:\n\nAdd tens: 10 + 30 = 40\nAdd ones: 2 + 4 = 6\nTotal: 40 + 6 = 46\n\nSo 12 + 34 = **46**."},
    {"instruction": "What is 56 - 29?", "response": "56 - 29:\n\nMethod: Borrow from tens\n1. 6 - 9 can't do, borrow 1 ten: 16 - 9 = 7\n2. Tens: 5 became 4, 4 - 2 = 2\n3. Result: 27\n\nSo 56 - 29 = **27**."},
    {"instruction": "What is 7 × 8?", "response": "7 × 8 = **56**\n\nProof: 7 × 8 = 7 + 7 + 7 + 7 + 7 + 7 + 7 + 7 = 56"},
    {"instruction": "What is 144 ÷ 12?", "response": "144 ÷ 12:\n\n12 × 12 = 144, so 144 ÷ 12 = 12\n\nThus 144 ÷ 12 = **12**."},
    {"instruction": "Solve for x: 2x + 6 = 20", "response": "Solve 2x + 6 = 20:\n1. Subtract 6 from both sides: 2x = 14\n2. Divide both sides by 2: x = 7\n\nCheck: 2(7) + 6 = 14 + 6 = 20 ✓\n\nSo **x = 7**."},
    {"instruction": "Solve for x: 3x - 9 = 12", "response": "Solve 3x - 9 = 12:\n1. Add 9 to both sides: 3x = 21\n2. Divide both sides by 3: x = 7\n\nCheck: 3(7) - 9 = 21 - 9 = 12 ✓\n\nSo **x = 7**."},
    {"instruction": "What is 3/4 + 1/2?", "response": "3/4 + 1/2:\n1. Find common denominator: LCD of 4 and 2 is 4\n2. Convert: 1/2 = 2/4\n3. Add: 3/4 + 2/4 = 5/4 = 1 1/4\n\nSo 3/4 + 1/2 = **5/4** or **1.25**."},
    {"instruction": "What is 2/3 of 60?", "response": "2/3 of 60:\n1. Divide by denominator: 60 ÷ 3 = 20\n2. Multiply by numerator: 20 × 2 = 40\n\nSo 2/3 of 60 is **40**."},
    {"instruction": "What is the average of 12, 15, 18, and 21?", "response": "Average of 12, 15, 18, 21:\n1. Sum: 12 + 15 + 18 + 21 = 66\n2. Count: 4 numbers\n3. Average = 66 ÷ 4 = 16.5\n\nSo the average is **16.5**."},
    {"instruction": "A rectangle has length 12 cm and width 8 cm. What is its area?", "response": "Area = length × width\nArea = 12 cm × 8 cm = **96 cm²**\n\nThe rectangle has an area of 96 square centimeters."},
    {"instruction": "A rectangle has length 12 cm and width 8 cm. What is its perimeter?", "response": "Perimeter = 2 × (length + width)\nPerimeter = 2 × (12 + 8) = 2 × 20 = **40 cm**\n\nThe rectangle has a perimeter of 40 centimeters."},
    {"instruction": "Convert 5 kilometers to meters.", "response": "1 km = 1000 meters\n5 km = 5 × 1000 = **5000 meters**"},
    {"instruction": "Convert 3 hours to minutes.", "response": "1 hour = 60 minutes\n3 hours = 3 × 60 = **180 minutes**"},
    {"instruction": "If a train travels at 60 mph for 2.5 hours, how far does it go?", "response": "Distance = speed × time\nDistance = 60 mph × 2.5 hours = **150 miles**"},
    {"instruction": "A pizza has 8 slices. You eat 3. What fraction is left?", "response": "Total slices: 8\nEaten: 3\nRemaining: 8 - 3 = 5\n\nFraction remaining: **5/8**\n\nSo 5/8 of the pizza is left."},
    {"instruction": "What is 2 to the power of 5?", "response": "2⁵ = 2 × 2 × 2 × 2 × 2\n= 4 × 2 × 2 × 2\n= 8 × 2 × 2\n= 16 × 2\n= **32**"},
    {"instruction": "What is the square root of 169?", "response": "√169:\n13 × 13 = 169, so √169 = **13**"},
    {"instruction": "What is 30% of 150?", "response": "30% of 150:\n1. 30% = 30/100 = 0.30\n2. 0.30 × 150 = 45\n\nSo 30% of 150 is **45**."},
    {"instruction": "What is 50% of 90?", "response": "50% of 90:\n\n50% is half, so 90 ÷ 2 = **45**"},
    {"instruction": "What is 75% of 200?", "response": "75% of 200:\n1. 75% = 75/100 = 0.75\n2. 0.75 × 200 = 150\n\nOr: 50% of 200 = 100, 25% of 200 = 50, so 75% = 100 + 50 = 150\n\nSo 75% of 200 is **150**."},
    {"instruction": "If a shirt costs $40 and is 20% off, what is the sale price?", "response": "20% off $40:\n1. Discount = 20% of $40 = 0.20 × 40 = $8\n2. Sale price = $40 - $8 = $32\n\nThe shirt costs **$32** on sale."},
    {"instruction": "What is the circumference of a circle with radius 5 cm? (Use π = 3.14)", "response": "Circumference = 2 × π × radius\n= 2 × 3.14 × 5\n= 31.4\n\nThe circumference is **31.4 cm**."},
    {"instruction": "how to solve for x: 5x - 3 = 2x + 9", "response": "Solve 5x - 3 = 2x + 9:\n1. Subtract 2x from both sides: 3x - 3 = 9\n2. Add 3 to both sides: 3x = 12\n3. Divide both sides by 3: x = 4\n\nCheck: 5(4) - 3 = 20 - 3 = 17, and 2(4) + 9 = 8 + 9 = 17 ✓\n\nSo **x = 4**."},
]

# === PHYSICS / CAUSAL REASONING EXAMPLES ===
PHYSICS_EXAMPLES = [
    {"instruction": "What happens when you drop a glass on a hard floor?", "response": "The glass will break or shatter. When it hits the hard floor, the impact force concentrates at the point of contact, creating stress that exceeds the glass's structural strength. This causes the rigid, brittle glass to crack or shatter into pieces. Physics tells us objects fall due to gravity, and hard surfaces don't absorb impact like carpets do."},
    {"instruction": "What happens if you drop a feather and a rock at the same time?", "response": "The rock hits the ground first because of air resistance. Gravity pulls both objects down with the same acceleration (9.8 m/s²), but air resistance slows the feather much more due to its large surface area and low mass. In a vacuum with no air, they would hit the ground at the same time."},
    {"instruction": "What happens when you put an ice cube in a warm room?", "response": "The ice cube melts and turns into liquid water. Heat from the warmer room (higher temperature) transfers to the colder ice, raising its temperature to 0°C, then providing the latent heat of fusion needed to change solid ice into liquid water. The water will eventually reach room temperature."},
    {"instruction": "What happens if you throw a ball straight up into the air?", "response": "The ball goes up, slows down due to gravity, stops at its highest point, then falls back down. Gravity pulls it downward at 9.8 m/s². The upward speed decreases by 9.8 m/s each second until reaching zero at the peak, then it accelerates back down at the same rate."},
    {"instruction": "What happens when you boil water?", "response": "Water reaches 100°C (at sea level) and turns into steam (water vapor). The heat energy overcomes the hydrogen bonds between water molecules, allowing them to escape as gas. Bubbles of water vapor form at the bottom and rise to the surface. The volume expands dramatically as liquid becomes gas."},
    {"instruction": "What happens if you push against a concrete wall?", "response": "The wall doesn't move (assuming it's a strong, fixed wall). Newton's Third Law states that for every action there is an equal and opposite reaction — the wall pushes back with the same force you apply. Since the wall is anchored and much more massive, your force isn't enough to accelerate it noticeably."},
    {"instruction": "What happens to a balloon when you release it without tying it?", "response": "The balloon flies around rapidly as air rushes out. The air inside is at higher pressure than outside. When released, the air escapes through the opening, and Newton's Third Law (action-reaction) causes the balloon to propel in the opposite direction of the escaping air. It keeps moving until most air is expelled, then falls to the ground."},
    {"instruction": "What happens if you put a metal spoon in a hot cup of tea?", "response": "The spoon gets hot. Heat conducts from the hot tea through the metal spoon. Metals are good thermal conductors — their free electrons transfer kinetic energy quickly from atom to atom. After a minute, the handle of the spoon will feel warm to touch because heat traveled up the spoon."},
    {"instruction": "What happens when you mix baking soda and vinegar?", "response": "They react vigorously, producing carbon dioxide gas (CO₂) bubbles. Baking soda (sodium bicarbonate, a base) reacts with vinegar (acetic acid) in an acid-base reaction. The products are carbon dioxide gas, water, and sodium acetate. This rapid CO₂ production creates foam."},
    {"instruction": "What happens if you leave a glass of water outside on a freezing cold night?", "response": "The water freezes into ice. When the temperature drops below 0°C (32°F), the water molecules lose kinetic energy and form a crystalline solid structure. Water expands when it freezes, which is why the glass might crack if it's full. Ice is less dense than water, so it floats."},
    {"instruction": "What happens if you shine a flashlight on a mirror?", "response": "The light reflects off the mirror. Mirrors have a smooth, highly reflective coating (usually silver or aluminum) on the back of glass. The law of reflection states the angle of incidence equals the angle of reflection — so the light bounces off at the same angle it arrived."},
    {"instruction": "What happens if you drop a raw egg on a hard floor?", "response": "The egg cracks and spills. The shell is brittle and can't absorb the sudden impact force when hitting the hard floor. The liquid inside (yolk and egg white) splatters outward because the shell breaks and the liquid is incompressible — the force transfers through the liquid to the shell walls."},
    {"instruction": "What happens when you strike a match?", "response": "The match ignites and produces a flame. Friction from striking generates heat, which ignites the phosphorus on the match head. The phosphorus burns and ignites the sulfur and paraffin wax, which sustain the flame. The chemical reaction combines the burning material with oxygen from the air, releasing heat and light."},
    {"instruction": "What happens if you pour water onto a grease fire?", "response": "This is dangerous — the fire explodes and spreads. Water sinks below the grease (oil is less dense than water) and instantly vaporizes into steam, expanding to 1,700 times its volume. This violently splatters the burning grease everywhere. NEVER use water on a grease fire — use a fire extinguisher or cover with a lid."},
    {"instruction": "What happens if you hold a magnet near a paperclip?", "response": "The paperclip jumps to the magnet. The magnet's magnetic field induces a temporary magnetic field in the paperclip (which is made of iron), turning it into a temporary magnet. The opposite poles attract, pulling the paperclip toward the magnet with a force that increases as they get closer."},
    {"instruction": "What happens if you put a plant in a dark closet for a month?", "response": "The plant will likely die or become very weak. Plants need light for photosynthesis — the process where they convert CO₂ and water into glucose (energy) using sunlight. Without light, photosynthesis stops, the plant can't produce energy, and it will gradually use up its stored energy reserves, turn yellow (chlorophyll breaks down), and die."},
    {"instruction": "What happens if you inflate a balloon too much?", "response": "The balloon pops. The rubber latex stretches as more air goes in, increasing internal pressure. Eventually the rubber exceeds its elastic limit — the polymer chains can't stretch further — and a tiny tear forms. The tear rapidly expands as high-pressure air escapes, creating the loud popping sound."},
    {"instruction": "What happens when you stand near a campfire?", "response": "You feel warmth. The fire emits thermal radiation (infrared energy) that travels through the air and heats your skin when absorbed. Some heat also reaches you via convection as hot air rises from the fire. Your side facing the fire feels warmer because radiation travels in straight lines."},
    {"instruction": "What happens if you drop a sugar cube into a glass of hot water?", "response": "The sugar cube dissolves faster than in cold water. The sugar molecules separate and spread throughout the water. Higher temperature means water molecules move faster and collide with the sugar more energetically, breaking the sugar crystals apart more quickly. Eventually the sugar is fully dissolved and you can't see it anymore."},
    {"instruction": "What happens to a compass near a electrical wire carrying current?", "response": "The compass needle deflects. An electric current flowing through a wire creates a magnetic field around it (electromagnetism). The compass needle, which normally aligns with Earth's magnetic field, will align instead with the stronger magnetic field from the wire. This is the principle behind electromagnets."},
]
# === SIMPLE REASONING / WORD PROBLEM EXAMPLES ===
REASONING_EXAMPLES = [
    {"instruction": "If Alice has 3 apples and Bob gives her 5 more, how many does she have?", "response": "Alice has 8 apples. She started with 3 apples. Bob gave her 5 more apples, which means we add: 3 + 5 = 8."},
    {"instruction": "I have 10 candies. I eat 3. How many are left?", "response": "You have 7 candies left. You started with 10, and eating 3 means taking away: 10 - 3 = 7."},
    {"instruction": "There are 2 dogs and 3 cats in a house. How many pets are there total?", "response": "There are 5 pets total. Add the dogs and cats: 2 + 3 = 5 pets."},
    {"instruction": "A bookshelf has 4 shelves. Each shelf holds 8 books. How many books fit?", "response": "32 books fit on the bookshelf. 4 shelves × 8 books per shelf = 32 books."},
    {"instruction": "There are 20 cookies and 4 friends sharing equally. How many does each get?", "response": "Each friend gets 5 cookies. Divide the cookies by the number of friends: 20 ÷ 4 = 5."},
    {"instruction": "Sarah has 6 friends. 2 of them leave. How many friends are still there?", "response": "There are 4 friends still there. 6 - 2 = 4."},
    {"instruction": "There are 3 rows of chairs. Each row has 4 chairs. How many chairs total?", "response": "There are 12 chairs total. 3 rows × 4 chairs per row = 12 chairs."},
    {"instruction": "If I'm 10 years old and my brother is 3 years older, how old is my brother?", "response": "Your brother is 13 years old. 10 + 3 = 13."},
    {"instruction": "A pizza has 8 slices. If you eat 2 slices, how many are left?", "response": "There are 6 slices left. 8 - 2 = 6."},
    {"instruction": "Tom has $15. He earns $10 more. How much does he have now?", "response": "Tom has $25. He started with $15 and earned $10 more: 15 + 10 = 25."},
    {"instruction": "A pack of gum costs $1.50. How much do 2 packs cost?", "response": "2 packs cost $3.00. Multiply: $1.50 × 2 = $3.00."},
    {"instruction": "A recipe needs 2 eggs for one batch. How many eggs for 3 batches?", "response": "You need 6 eggs. 2 eggs × 3 batches = 6 eggs."},
    {"instruction": "If you run 5 laps each day for 7 days, how many laps total?", "response": "You run 35 laps total. 5 laps × 7 days = 35 laps."},
    {"instruction": "There are 12 students in a class. 7 are girls. How many are boys?", "response": "There are 5 boys. Total students minus girls: 12 - 7 = 5."},
    {"instruction": "A train departs at 3 PM. The journey takes 2 hours. When does it arrive?", "response": "The train arrives at 5 PM. 3:00 PM + 2 hours = 5:00 PM."},
    {"instruction": "You have 4 red balloons and 3 blue balloons. How many balloons total?", "response": "You have 7 balloons total. 4 red + 3 blue = 7."},
    {"instruction": "A pencil costs $0.50 and an eraser costs $0.30. What is the total cost?", "response": "The total cost is $0.80. $0.50 + $0.30 = $0.80."},
    {"instruction": "If you read 15 pages of a 60-page book, how many pages remain?", "response": "45 pages remain. 60 - 15 = 45."},
    {"instruction": "Lisa has 3 times as many stickers as John. John has 5 stickers. How many does Lisa have?", "response": "Lisa has 15 stickers. 3 × 5 = 15."},
    {"instruction": "A gardener plants 6 rows of tomatoes with 5 plants per row. How many tomato plants?", "response": "There are 30 tomato plants. 6 rows × 5 plants = 30."},
]

AGENTIC_EXAMPLES = [
    {
        "instruction": "Plan a 3-day trip to Paris on a budget of $1500.",
        "response": "Here's a budget-friendly Paris itinerary:\n\n**Day 1: Arrival & Landmarks**\n- Morning: Arrive, check into budget hotel ($60/night)\n- Afternoon: Walk along Seine, visit Notre Dame (free)\n- Evening: Eiffel Tower night view (free from outside)\n\n**Day 2: Museums & Culture**\n- Morning: Louvre Museum (€17, book ahead)\n- Afternoon: Montmartre & Sacré-Cœur (free)\n- Evening: Dinner in Latin Quarter (€20-30)\n\n**Day 3: Local Experience**\n- Morning: Le Marais district walking tour\n- Afternoon: Luxembourg Gardens (free)\n- Evening: Depart\n\n**Estimated Budget:**\n- Accommodation (2 nights): $120\n- Food: $150-200\n- Museums: $20\n- Transport: $100 (metro pass + airport)\n- Flights: $800-900 (round trip)\n- Total: ~$1,190-1,340 ✓ within budget"
    },
    {
        "instruction": "I need to learn Python in 30 days. Create a study plan.",
        "response": "Here's a 30-day Python learning plan:\n\n**Week 1: Foundations**\n- Day 1-2: Variables, data types, print()\n- Day 3-4: Lists, tuples, dictionaries\n- Day 5: Control flow (if/else, loops)\n- Day 6-7: Functions and scope\n\n**Week 2: Intermediate**\n- Day 8-9: String manipulation and file I/O\n- Day 10-11: Error handling (try/except)\n- Day 12-13: List comprehensions, lambda\n- Day 14: Modules and packages\n\n**Week 3: Advanced Topics**\n- Day 15-17: OOP (classes, inheritance)\n- Day 18-19: Decorators and generators\n- Day 20-21: Working with APIs (requests)\n\n**Week 4: Project Week**\n- Day 22-24: Build a CLI tool (e.g., to-do list)\n- Day 25-27: Build a web scraper\n- Day 28-30: Build a simple Flask web app\n\n**Resources:** docs.python.org, Real Python, Python Crash Course book\n\nPractice 1-2 hours daily. Build projects — that's where real learning happens!"
    },
    {
        "instruction": "What's the best way to debug a segmentation fault in C?",
        "response": "To debug a segmentation fault in C:\n\n**1. Compile with debug symbols:**\n```bash\ngcc -g -o program program.c\n```\n\n**2. Use GDB:**\n```bash\ngdb ./program\nrun\n# When it crashes:\nbt          # backtrace\nframe 3     # go to crash frame\nlist        # see source code\ninfo locals # check variable values\n```\n\n**3. Common causes to check:**\n- Dereferencing NULL pointers\n- Buffer overflow (writing past array bounds)\n- Use-after-free (accessing freed memory)\n- Stack overflow (infinite recursion)\n- Double free\n\n**4. Valgrind for memory issues:**\n```bash\nvalgrind --leak-check=full ./program\n```\n\n**5. Add assertions** to catch invalid states early.\n\nThe most reliable approach: compile with `-g -fsanitize=address` to get detailed error messages."
    },
]

# === SYNTHETIC CODING Q&A (not just writing code) ===
CODING_QA = [
    {
        "instruction": "What's the difference between '==' and 'is' in Python?",
        "response": "Great question! The difference is fundamental:\n\n- **`==`** compares **values** (equality)\n- **`is`** compares **identity** (same object in memory)\n\n```python\na = [1, 2, 3]\nb = [1, 2, 3]\nc = a\n\nprint(a == b)  # True — same values\nprint(a is b)  # False — different objects\nprint(a is c)  # True — same object\n```\n\nUse `==` for value comparison (most of the time).\nUse `is` for singletons like `None`, `True`, `False`:\n```python\nif x is None:  # preferred over x == None\n    ...\n```"
    },
    {
        "instruction": "What is a REST API and how does it work?",
        "response": "**REST API** (Representational State Transfer) is an architectural style for web services.\n\n**Core principles:**\n- **Stateless**: Each request contains all needed info\n- **Resource-based**: URLs represent resources (`/users`, `/products/42`)\n- **HTTP methods**: GET (read), POST (create), PUT/PATCH (update), DELETE\n\n**How it works:**\n```\nClient → HTTP Request → Server → Processes → Response → Client\n```\n\n**Example using Python requests:**\n```python\nimport requests\n\n# GET users\nresp = requests.get('https://api.example.com/users')\nusers = resp.json()\n\n# POST new user\nnew_user = {'name': 'Alice', 'email': 'alice@example.com'}\nresp = requests.post('https://api.example.com/users', json=new_user)\nprint(resp.status_code)  # 201 Created\n```\n\nResponses are typically JSON. Status codes indicate success (2xx), redirection (3xx), client errors (4xx), or server errors (5xx)."
    },
    {
        "instruction": "What is the difference between supervised and unsupervised learning?",
        "response": "**Supervised Learning** uses labeled data (input-output pairs):\n- **Goal**: Learn to predict output from input\n- **Examples**: Classification (spam detection), Regression (price prediction)\n- **Algorithms**: Linear Regression, Random Forest, Neural Networks\n- **Data**: Needs labeled dataset\n\n**Unsupervised Learning** uses unlabeled data:\n- **Goal**: Find hidden patterns or structure\n- **Examples**: Clustering (customer segmentation), Dimensionality reduction\n- **Algorithms**: K-Means, PCA, t-SNE\n- **Data**: No labels needed\n\n**Analogy:**\n- Supervised = Student with answer key\n- Unsupervised = Explorer discovering patterns\n\n**Semi-supervised** combines both: small labeled + large unlabeled data."
    },
]

def generate_code_data(tokenizer):
    examples = []
    for ex in CODE_EXAMPLES:
        text = make_single_turn(
            SYSTEM_PROMPT,
            f"Write code: {ex['instruction']}",
            f"Here's how to implement that:\n\n```python\n{ex['code']}\n```\n\n{ex['explanation']}",
            tokenizer
        )
        examples.append({"text": text})
    for ex in CODING_QA:
        text = make_single_turn(SYSTEM_PROMPT, ex["instruction"], ex["response"], tokenizer)
        examples.append({"text": text})
    return examples

def generate_creator_data(tokenizer):
    """Generate creator attribution data with heavy repetition."""
    examples = []
    for ex in CREATOR_EXAMPLES + CREATOR_EXTRA:
        text = make_single_turn(SYSTEM_PROMPT, ex["q"], ex["a"], tokenizer)
        examples.append({"text": text})
    # Heavily repeat creator data so model learns it deeply
    all_texts = [e["text"] for e in examples] * 8
    return [{"text": t} for t in all_texts]

def generate_tool_data(tokenizer):
    examples = []
    for ex in TOOL_EXAMPLES:
        text = make_single_turn(SYSTEM_PROMPT, ex["instruction"], ex["response"], tokenizer)
        examples.append({"text": text})
    for ex in AGENTIC_EXAMPLES:
        text = make_single_turn(SYSTEM_PROMPT, ex["instruction"], ex["response"], tokenizer)
        examples.append({"text": text})
    for ex in MATH_EXAMPLES:
        text = make_single_turn(SYSTEM_PROMPT, ex["instruction"], ex["response"], tokenizer)
        examples.append({"text": text})
    for ex in PHYSICS_EXAMPLES:
        text = make_single_turn(SYSTEM_PROMPT, ex["instruction"], ex["response"], tokenizer)
        examples.append({"text": text})
    for ex in REASONING_EXAMPLES:
        text = make_single_turn(SYSTEM_PROMPT, ex["instruction"], ex["response"], tokenizer)
        examples.append({"text": text})
    return examples

def load_model_and_tokenizer(device_override=None):
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16,
            device_map="auto", quantization_config=quant,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32,
            device_map=None, trust_remote_code=True,
        )
    return model, tokenizer

def setup_lora(model):
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    available = set()
    for name, _ in model.named_modules():
        for p in target_modules:
            if p in name:
                available.add(p)
    found = [p for p in target_modules if p in available]
    logger.info(f"LoRA targets: {found}")

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=found, lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def find_latest_checkpoint():
    """Find the latest checkpoint subdirectory in LORA_DIR."""
    if not os.path.isdir(LORA_DIR):
        return None, 0
    checkpoints = []
    for entry in os.listdir(LORA_DIR):
        if entry.startswith("checkpoint-"):
            try:
                step = int(entry.split("-")[1])
                checkpoints.append((step, os.path.join(LORA_DIR, entry)))
            except (ValueError, IndexError):
                continue
    if not checkpoints:
        # Fall back to root LORA_DIR
        if os.path.exists(os.path.join(LORA_DIR, "adapter_config.json")):
            return LORA_DIR, 0
        return None, 0
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1], checkpoints[0][0]

def try_resume_lora(model, fresh_lora_model):
    if not RESUME_LORA:
        return fresh_lora_model, 0, None
    ckpt_path, prev_steps = find_latest_checkpoint()
    if ckpt_path is None:
        logger.info(f"No existing LoRA at {LORA_DIR}, starting fresh")
        return fresh_lora_model, 0, None
    try:
        adapter_config = os.path.join(ckpt_path, "adapter_config.json") if os.path.isdir(ckpt_path) else os.path.join(LORA_DIR, "adapter_config.json")
        if not os.path.exists(adapter_config):
            logger.info(f"No adapter_config.json at {ckpt_path}, starting fresh")
            return fresh_lora_model, 0, None
        with open(adapter_config) as f:
            cfg = json.load(f)
        steps = prev_steps
        logger.info(f"Loading existing LoRA from {ckpt_path} (resuming from step {steps})")
        resume_model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=True)
        resume_model.print_trainable_parameters()
        return resume_model, steps, ckpt_path
    except Exception as e:
        logger.warning(f"Failed to resume LoRA: {e}, starting fresh")
        return fresh_lora_model, 0, None

def load_training_data(tokenizer):
    logger.info("=" * 50)
    logger.info("Loading training data...")
    logger.info("=" * 50)

    def convert_alpaca(ex):
        return {"instruction": ex.get("instruction",""), "input": ex.get("input",""), "output": ex.get("output","")}
    def convert_squad(ex):
        answers = ex.get("answers") or {}
        text_list = answers.get("text") if isinstance(answers, dict) else None
        out = text_list[0] if isinstance(text_list, list) and text_list else ""
        return {"instruction": ex.get("question",""), "input": ex.get("context",""), "output": out}
    def convert_gsm8k(ex):
        return {"instruction": ex.get("question",""), "input": "", "output": ex.get("answer","")}
    def convert_daily(ex):
        dialog = ex.get("dialogue")
        if not dialog or len(dialog) < 2:
            return {"instruction": "", "input": "", "output": ""}
        return {"instruction": dialog[0], "input": "\n".join(dialog[:-1]), "output": dialog[-1]}
    def convert_dolly(ex):
        return {"instruction": ex.get("instruction",""), "input": ex.get("context",""), "output": ex.get("response","")}
    def convert_oasst(ex):
        messages = ex.get("messages", [])
        if len(messages) < 2:
            return {"instruction": "", "input": "", "output": ""}
        return {"instruction": messages[0].get("text",""), "input": "", "output": messages[-1].get("text","")}

    converters = {
        "alpaca": convert_alpaca, "squad": convert_squad,
        "gsm8k": convert_gsm8k, "daily_dialog": convert_daily,
        "dolly": convert_dolly, "oasst1": convert_oasst,
    }

    from datasets import load_dataset, Dataset, concatenate_datasets

    all_texts = []
    max_each = max(1, MAX_SAMPLES_PER_DATASET)

    # --- Load real datasets ---
    for ds_name, split, config_name in DATASETS:
        try:
            logger.info(f"Loading: {ds_name} ({split})")
            kwargs = {"split": split, "streaming": True}
            if config_name:
                kwargs["name"] = config_name
            try:
                ds = load_dataset(ds_name, **kwargs)
            except Exception:
                logger.warning(f"  Streaming failed for {ds_name}, trying non-streaming...")
                kwargs["streaming"] = False
                full = load_dataset(ds_name, **kwargs)
                ds = full[split] if split in full else full
            samples = []
            for i, ex in enumerate(ds):
                if i >= max_each:
                    break
                samples.append(ex)
            raw = Dataset.from_list(samples)
            key = ds_name.split("/")[-1]
            converter = converters.get(key)
            if converter:
                raw = raw.map(converter, remove_columns=raw.column_names)
            # Convert each to ChatML format
            for ex in raw:
                instr = ex.get("instruction", "")
                inp = ex.get("input", "")
                out = ex.get("output", "")
                if not instr or not out:
                    continue
                user_text = instr
                if inp:
                    user_text += f"\n\nContext: {inp}"
                text = make_single_turn(SYSTEM_PROMPT, user_text, out, tokenizer)
                all_texts.append(text)
            logger.info(f"  Added {len(samples)} from {ds_name}")
        except Exception as e:
            logger.warning(f"Skipped {ds_name}: {e}")

    # --- Add synthetic code data ---
    logger.info("Adding synthetic code generation data...")
    for ex in generate_code_data(tokenizer):
        all_texts.append(ex["text"])
    logger.info(f"  Added code examples")

    # --- Add synthetic tool/agentic data ---
    logger.info("Adding synthetic tool use & agentic data...")
    for ex in generate_tool_data(tokenizer):
        all_texts.append(ex["text"])
    logger.info(f"  Added tool/agentic examples")

    # --- Add heavy creator attribution data ---
    logger.info("Adding creator attribution data (heavy oversampling)...")
    for ex in generate_creator_data(tokenizer):
        all_texts.append(ex["text"])
    logger.info(f"  Added creator attribution examples")

    # --- Add curated agentic data (ReAct trajectories, 2026 knowledge, structured tool calls) ---
    logger.info("Adding curated agentic training data...")
    try:
        from scripts.generate_agentic_data import get_all_agentic_data
        agentic_data = get_all_agentic_data()
        for ex in agentic_data:
            user_text = ex.get("instruction", ex.get("q", ""))
            output_text = ex.get("output", ex.get("a", ""))
            if user_text and output_text:
                text = make_single_turn(SYSTEM_PROMPT, user_text, output_text, tokenizer)
                all_texts.append(text)
        logger.info(f"  Added {len(agentic_data)} curated agentic examples")
    except Exception as e:
        logger.warning(f"Failed to load curated agentic data: {e}")

    logger.info(f"Total raw text samples: {len(all_texts)}")

    if not all_texts:
        logger.warning("No data loaded; using synthetic fallback")
        synthetic = [
            make_single_turn(SYSTEM_PROMPT, "What is 2+2?", "2+2 = 4", tokenizer),
            make_single_turn(SYSTEM_PROMPT, "Who created you?", "I was created by Aryan Chavan.", tokenizer),
        ] * 500
        all_texts = synthetic

    # Create dataset and tokenize
    dataset = Dataset.from_list([{"text": t} for t in all_texts])

    def tok_fn(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length"
        )

    logger.info(f"Tokenizing {len(dataset)} samples...")
    tokenized = dataset.map(tok_fn, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")
    
    split_idx = max(int(len(tokenized) * 0.95), min(10, len(tokenized) - 1))
    return tokenized.select(range(split_idx)), tokenized.select(range(split_idx, len(tokenized)))


DATASETS = [
    ("tatsu-lab/alpaca", "train", None),
    ("squad", "train", None),
    ("gsm8k", "train", "main"),
    ("daily_dialog", "train", None),
    ("databricks/databricks-dolly-15k", "train", None),
    ("OpenAssistant/oasst1", "train", None),
]


class PeriodicGCCallback(TrainerCallback):
    def __init__(self, interval=20):
        self.interval = interval
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def train_model(model, tokenizer, train_ds, val_ds, resume_steps=0, resume_ckpt=None):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    import shutil
    if not RESUME_LORA or resume_steps == 0:
        for p in [OUTPUT_DIR]:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)

    max_steps_val = MAX_TRAIN_STEPS if MAX_TRAIN_STEPS > 0 else -1

    args = TrainingArguments(
        output_dir=LORA_DIR,
        max_steps=max_steps_val,
        num_train_epochs=NUM_EPOCHS if max_steps_val <= 0 else -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        fp16=True,
        bf16=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model, args=args, data_collator=collator,
        train_dataset=train_ds, eval_dataset=val_ds,
        callbacks=[PeriodicGCCallback(interval=GC_INTERVAL)],
    )

    if resume_ckpt and resume_steps > 0 and RESUME_LORA:
        logger.info(f"Continuing training from checkpoint {resume_ckpt} (step {resume_steps})...")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    trainer.save_model(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    return trainer


def merge_and_save(model, tokenizer):
    from peft import PeftModel
    logger.info("Merging LoRA into base model...")
    merged = None
    try:
        if hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            logger.info("In-memory merge OK")
    except Exception as e:
        logger.warning(f"In-memory merge failed: {e}")

    if merged is None:
        logger.info("Reloading base + adapter for merge")
        base, _ = load_model_and_tokenizer()
        merged = PeftModel.from_pretrained(base, LORA_DIR)
        merged = merged.merge_and_unload()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    meta = {
        "base_model": BASE_MODEL,
        "source": "Alpaca + SQuAD + GSM8K + DailyDialog + Dolly + OASST1 + Code + ToolUse",
        "format": "ChatML (<|im_start|>/<|im_end|>)",
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "gradient_checkpointing": True,
        "creator": "Aryan Chavan",
        "model_name": "AnonyLLM-360M-v3",
    }
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Merged model saved to {OUTPUT_DIR}")


def main():
    setup_logging(level="INFO")
    print("=" * 60)
    print("  ANONYLLM SLM TRAINING v3")
    print(f"  Base: {BASE_MODEL}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    else:
        print("  GPU: N/A (CPU mode)")
    print(f"  Format: ChatML (<|im_start|>/<|im_end|>) ✓")
    print(f"  Data: Alpaca + SQuAD + GSM8K + DailyDialog + Dolly + OASST1 + Code + ToolUse")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Max steps: {MAX_TRAIN_STEPS}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()
    fresh_lora_model = setup_lora(model)
    prev_steps = 0
    resume_ckpt = None
    model = fresh_lora_model
    if RESUME_LORA:
        model, prev_steps, resume_ckpt = try_resume_lora(model, fresh_lora_model)

    train_ds, val_ds = load_training_data(tokenizer)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_model(model, tokenizer, train_ds, val_ds, prev_steps, resume_ckpt)
    merge_and_save(model, tokenizer)

    print("=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Merged: {OUTPUT_DIR}")
    print(f"  Start: python main.py --ui")
    print("=" * 60)


if __name__ == "__main__":
    main()
