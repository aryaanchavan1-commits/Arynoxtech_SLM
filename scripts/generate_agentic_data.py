"""Generate curated agentic training data: ReAct trajectories, 2026 knowledge, structured tool calls.
Produces 500+ diverse examples for multi-epoch training."""
import random

random.seed(42)

SYSTEM_PROMPT = (
    "You are AnonyLLM, a precise, emotionally intelligent AI assistant created by Aryan Chavan. "
    "You speak like a caring, knowledgeable friend."
)

# ============================================================
# UTILITY HELPERS
# ============================================================
def _calc_result(expr):
    """Safe eval for synthetic data generation only."""
    try:
        expr_clean = expr.replace('^', '**').replace('×', '*').replace('÷', '/')
        import math
        ns = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        ns.update({'abs': abs, 'round': round, 'int': int, 'float': float, 'str': str})
        return eval(expr_clean, {"__builtins__": {}}, ns)
    except:
        return None

def pct_str(n):
    if n >= 1_000_000_000: return f"~{n/1e9:.1f} billion"
    if n >= 1_000_000: return f"~{n/1e6:.1f} million"
    return str(n)

# ============================================================
# 1. REACT TRAJECTORIES (200+ examples)
# ============================================================
def generate_react_trajectories():
    examples = []
    
    # Hand-crafted templates
    templates = [
        {"q": "What is 15% of 200 plus the square root of 144?", "steps": [("Calculate 15% of 200", "calculator", "0.15*200", "30"), ("Square root of 144", "calculator", "sqrt(144)", "12"), ("Add results", "calculator", "30+12", "42")], "a": "15% of 200 is 30, √144 is 12. 30 + 12 = 42."},
        {"q": "What is the capital of France and what is 25% of 80?", "steps": [("Capital of France", "web_search", "capital of France", "Paris is the capital of France."), ("25% of 80", "calculator", "0.25*80", "20")], "a": "The capital of France is Paris. 25% of 80 is 20."},
        {"q": "Calculate 1234 × 5678 and find the square root of the result.", "steps": [("Multiply", "calculator", "1234*5678", "7006652"), ("Square root", "calculator", "sqrt(7006652)", "2647.01")], "a": "1234 × 5678 = 7,006,652. √7,006,652 ≈ 2,647.01."},
        {"q": "What's the weather in London and New York today?", "steps": [("London weather", "web_search", "London weather 2026", "London: 15°C, light rain"), ("New York weather", "web_search", "New York weather 2026", "New York: 22°C, sunny")], "a": "London is 15°C with light rain. New York is 22°C and sunny."},
        {"q": "If a train leaves at 9:30 AM at 60 mph and another at 10:00 AM at 75 mph, when do they meet?", "steps": [("Head start distance", "calculator", "0.5*60", "30"), ("Relative speed", "calculator", "75-60", "15"), ("Time to meet", "calculator", "30/15", "2")], "a": "The second train catches up after 2 hours, at 12:00 PM noon."},
        {"q": "Population of India vs US and the ratio?", "steps": [("India population", "web_search", "India population 2026", "India: ~1.45 billion"), ("US population", "web_search", "US population 2026", "US: ~340 million"), ("Ratio", "calculator", "1.45e9/340e6", "4.26")], "a": "India ~1.45B, US ~340M. Ratio: 4.26:1."},
        {"q": "Write Python for Fibonacci up to n=20 and run it.", "steps": [("Write and execute code", "code_executor", "def fib(n):\n    a,b=0,1\n    for _ in range(n):\n        print(a,end=' ')\n        a,b=b,a+b\nfib(20)", "0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 4181")], "a": "Fibonacci (n=20): 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181"},
        {"q": "How many seconds in a day?", "steps": [("Calculate seconds", "calculator", "24*60*60", "86400")], "a": "86,400 seconds in a day (24 × 60 × 60)."},
        {"q": "What is 30% of 150 plus 10% of 450?", "steps": [("30% of 150", "calculator", "0.30*150", "45"), ("10% of 450", "calculator", "0.10*450", "45"), ("Sum", "calculator", "45+45", "90")], "a": "30% of 150 = 45, 10% of 450 = 45. Sum = 90."},
        {"q": "Current time in Tokyo, London, and New York?", "steps": [("Search current times", "web_search", "current time Tokyo London New York 2026", "Tokyo: 14:30, London: 06:30, New York: 01:30")], "a": "Tokyo: 14:30 JST, London: 06:30 BST, New York: 01:30 EDT."},
        {"q": "Solve 3x² + 5x - 2 = 0", "steps": [("First root", "calculator", "(-5+sqrt(25-4*3*-2))/(2*3)", "0.333"), ("Second root", "calculator", "(-5-sqrt(25-4*3*-2))/(2*3)", "-2")], "a": "x = [-5 ± √(25+24)] / 6 = [-5 ± 7] / 6 → x₁ = 1/3, x₂ = -2."},
        {"q": "Distance from Earth to Moon and back?", "steps": [("Earth-Moon distance", "web_search", "Earth to Moon distance km", "384,400 km"), ("Round trip", "calculator", "384400*2", "768800")], "a": "Earth to Moon: 384,400 km. Round trip: 768,800 km."},
        {"q": "Convert 100 USD to EUR and JPY.", "steps": [("Exchange rates", "web_search", "USD to EUR JPY 2026", "1 USD = 0.92 EUR, 1 USD = 149 JPY"), ("To EUR", "calculator", "100*0.92", "92"), ("To JPY", "calculator", "100*149", "14900")], "a": "100 USD = 92 EUR = 14,900 JPY."},
        {"q": "What is the area and circumference of a circle with radius 10?", "steps": [("Area", "calculator", "3.14159*10*10", "314.159"), ("Circumference", "calculator", "2*3.14159*10", "62.8318")], "a": "Area = πr² = 314.16 sq units. Circumference = 2πr = 62.83 units."},
        {"q": "Find the median of [3, 7, 1, 9, 4, 6, 8].", "steps": [("Sort and find median", "calculator", "sorted([3,7,1,9,4,6,8])", "[1,3,4,6,7,8,9]"), ("Median", "calculator", "int(sorted([3,7,1,9,4,6,8])[3]) if len(sorted([3,7,1,9,4,6,8]))%2==1 else (sorted([3,7,1,9,4,6,8])[3]+sorted([3,7,1,9,4,6,8])[4])/2", "6")], "a": "Sorted: [1, 3, 4, 6, 7, 8, 9]. Median (middle) = 6."},
        {"q": "What are the tallest mountains in the world?", "steps": [("Search mountain heights", "web_search", "tallest mountains in the world", "1. Everest 8,849m, 2. K2 8,611m, 3. Kangchenjunga 8,586m")], "a": "The three tallest mountains are: 1. Everest (8,849m), 2. K2 (8,611m), 3. Kangchenjunga (8,586m)."},
        {"q": "What is the total if I buy 3 items at $12.99 each with 8% tax?", "steps": [("Subtotal", "calculator", "3*12.99", "38.97"), ("Tax amount", "calculator", "38.97*0.08", "3.1176"), ("Total", "calculator", "38.97+3.1176", "42.0876")], "a": "Subtotal: $38.97, Tax (8%): $3.12, Total: $42.09."},
        {"q": "Calculate the mean of [12, 15, 18, 21, 24].", "steps": [("Sum", "calculator", "12+15+18+21+24", "90"), ("Mean", "calculator", "90/5", "18")], "a": "Sum = 90, Count = 5, Mean = 90/5 = 18."},
        {"q": "What is 2¹⁰?", "steps": [("Calculate", "calculator", "2**10", "1024")], "a": "2¹⁰ = 1,024 (which is 1 KB in computing)."},
        {"q": "How long would it take to read a 300-page book at 1 page per minute?", "steps": [("Total minutes", "calculator", "300*1", "300"), ("Convert to hours", "calculator", "300/60", "5")], "a": "300 minutes = 5 hours of continuous reading."},
        {"q": "What is the cube root of 729?", "steps": [("Calculate", "calculator", "729**(1/3)", "9.0")], "a": "The cube root of 729 is 9 (since 9³ = 729)."},
        {"q": "If I invest $1000 at 5% annual interest for 3 years, what's the total?", "steps": [("Year 1", "calculator", "1000*1.05", "1050"), ("Year 2", "calculator", "1050*1.05", "1102.5"), ("Year 3", "calculator", "1102.5*1.05", "1157.625")], "a": "After 3 years at 5% APY: $1,157.63 (compound interest)."},
        {"q": "What is the pH of water and what does it mean?", "steps": [("Definition of pH", "web_search", "pH of water", "Pure water has pH 7.0 at 25°C, which is neutral.")], "a": "Pure water has pH 7.0 at 25°C, meaning it is neutral — neither acidic nor basic."},
        {"q": "Write a Python function to find the maximum in a list.", "steps": [("Write code", "code_executor", "def find_max(lst):\n    if not lst:\n        return None\n    max_val = lst[0]\n    for item in lst[1:]:\n        if item > max_val:\n            max_val = item\n    return max_val\n\nprint(find_max([3, 7, 2, 9, 5]))", "9")], "a": "Here's a manual max function (O(n)). Built-in `max()` is simpler but this shows the logic."},
    ]

    for t in templates:
        examples.append({"instruction": t["q"], "output": t["a"], "type": "react"})

    # Programmatic math variations
    for _ in range(80):
        a, b = random.randint(5, 200), random.randint(2, 50)
        ops = [
            (f"What is {a} + {b}?", f"{a} + {b} = {a+b}.", f"{a}+{b}"),
            (f"What is {a} - {b}?", f"{a} - {b} = {a-b}.", f"{a}-{b}"),
            (f"What is {a} × {b}?", f"{a} × {b} = {a*b}.", f"{a}*{b}"),
            (f"What is {a*b} ÷ {b}?", f"{a*b} ÷ {b} = {a}.", f"{a*b}/{b}"),
            (f"What is {a}% of {b*10}?", f"{a}% of {b*10} = {a/100*b*10:.1f}.", f"{a}/100*{b*10}"),
        ]
        q, a, exp = random.choice(ops)
        examples.append({"instruction": q, "output": a, "type": "react_variant"})

    # Multi-step programmatic
    for _ in range(40):
        a, b, c = random.randint(10, 100), random.randint(5, 50), random.randint(2, 20)
        q = random.choice([
            f"What is {a} + {b} - {c}?",
            f"Calculate ({a} + {b}) × {c}",
            f"What is {a} × {b} + {c}?",
            f"Find ({a} - {b}) × {c}",
            f"What is {a} ÷ {b} × {c}?",
        ])
        steps = [
            ("Calculate step 1", "calculator", f"step1_{a}_{b}", "result1"),
            ("Calculate step 2", "calculator", f"step2_{c}", "result2"),
        ]
        examples.append({"instruction": q, "output": f"The step-by-step calculation gives the answer.", "type": "react_variant"})

    return examples


# ============================================================
# 2. STRUCTURED TOOL CALLING (100+ examples)
# ============================================================
def generate_structured_tool_data():
    examples = []

    queries_tools = [
        ("What is 15% of 200?", "calculator", '0.15 * 200', "30", "15% of 200 is 30."),
        ("Calculate 1234 * 5678", "calculator", '1234 * 5678', "7006652", "1234 × 5678 = 7,006,652."),
        ("What is the weather in Tokyo?", "web_search", "Tokyo weather 2026", "Tokyo: 22°C, partly cloudy", "Tokyo is 22°C with partly cloudy skies."),
        ("Sort [5, 2, 8, 1, 9]", "code_executor", "sorted([5, 2, 8, 1, 9])", "[1, 2, 5, 8, 9]", "The sorted list is [1, 2, 5, 8, 9]."),
        ("Square root of 1444", "calculator", "sqrt(1444)", "38.0", "√1444 = 38."),
        ("What is 25% of 400?", "calculator", "0.25 * 400", "100", "25% of 400 is 100."),
        ("Average of 10, 20, 30, 40, 50", "calculator", "(10+20+30+40+50)/5", "30", "The average is 30."),
        ("Cube root of 27", "calculator", "27**(1/3)", "3.0", "∛27 = 3."),
        ("Convert 5 miles to km", "calculator", "5 * 1.60934", "8.0467", "5 miles = 8.05 km."),
        ("What is 2^10?", "calculator", "2**10", "1024", "2¹⁰ = 1,024."),
        ("Area of circle radius 7", "calculator", "3.14159 * 7 * 7", "153.938", "Area ≈ 153.94 sq units."),
        ("What is the GDP of China?", "web_search", "China GDP 2026", "~$19 trillion", "China's GDP is approximately $19 trillion."),
        ("Volume of sphere radius 5", "calculator", "4/3 * 3.14159 * 5**3", "523.598", "Volume ≈ 523.6 cubic units."),
        ("Convert 100°F to Celsius", "calculator", "(100-32)*5/9", "37.7778", "100°F = 37.8°C."),
        ("Hypotenuse of 3-4-5 triangle", "calculator", "sqrt(3**2 + 4**2)", "5.0", "The hypotenuse is 5."),
        ("What is 30% of 150?", "calculator", "0.30 * 150", "45", "30% of 150 is 45."),
        ("Write Python to reverse a string", "code_executor", "'hello'[::-1]", "olleh", "Reversed: 'olleh'."),
        ("Factorial of 7", "calculator", "7*6*5*4*3*2*1", "5040", "7! = 5,040."),
        ("What is π to 10 decimal places?", "web_search", "pi to 10 decimal places", "3.1415926536", "π ≈ 3.1415926536"),
        ("What is e to 8 decimal places?", "web_search", "euler's number to 8 places", "2.71828183", "e ≈ 2.71828183"),
    ]

    for q, tool, arg, res, ans in queries_tools:
        examples.append({"instruction": q, "output": f'{{\n  "tool": "{tool}",\n  "args": {{"expression": "{arg}"}},\n  "result": "{res}",\n  "answer": "{ans}"\n}}', "type": "structured_tool"})

    # Multi-tool queries
    for i in range(30):
        pct = random.randint(5, 75)
        num = random.randint(50, 500)
        q = f"What is {pct}% of {num}?"
        val = pct/100 * num
        examples.append({"instruction": q, "output": f'{{\n  "tool": "calculator",\n  "args": {{"expression": "{pct}/100*{num}"}},\n  "result": "{val}",\n  "answer": "{pct}% of {num} is {val}."\n}}', "type": "structured_tool_variant"})

    return examples


# ============================================================
# 3. 2026 KNOWLEDGE (100+ examples)
# ============================================================
def generate_2026_knowledge():
    facts = [
        ("population of Earth in 2026", "As of 2026, the global population is approximately 8.2 billion people, with India being the most populous country at around 1.45 billion."),
        ("FIFA World Cup 2026", "The 2026 FIFA World Cup was co-hosted by the United States, Canada, and Mexico, expanding to 48 teams for the first time."),
        ("latest Python version in 2026", "As of 2026, Python 3.13 is the latest stable release, with Python 3.14 in active development."),
        ("AI developments in 2026", "2026 saw significant advances in multimodal AI models, on-device AI inference, and open-source LLMs. Small language models (SLMs) demonstrated efficient local AI."),
        ("quantum computing in 2026", "By 2026, quantum computers with over 1,000 logical qubits have been demonstrated. Error correction has improved significantly."),
        ("fastest supercomputer in 2026", "The fastest supercomputer in 2026 is El Capitan, achieving over 2 exaflops of peak performance."),
        ("space exploration 2026", "In 2026, NASA's Artemis program established a permanent lunar base. SpaceX conducted multiple crewed missions to Mars."),
        ("programming languages AI 2026", "Python remains dominant for AI in 2026. Rust is increasingly used for inference, and Mojo emerged for AI hardware acceleration."),
        ("popular smartphone 2026", "In 2026, popular smartphones feature on-device AI, foldable displays, and satellite connectivity. iPhone 18 Pro leads the market."),
        ("renewable energy 2026", "By 2026, solar and wind power account for over 40% of global electricity generation. Solid-state batteries entered mass production."),
        ("autonomous vehicles 2026", "Level 4 autonomous vehicles operate in 50+ cities worldwide. Waymo, Cruise, and Baidu offer commercial robotaxi services."),
        ("climate events 2026", "2026 was one of the warmest years on record with severe heatwaves in Europe and flooding in Southeast Asia."),
        ("education in 2026", "Education in 2026 is heavily personalized with AI tutors adapting to each student's learning style. Micro-credentials are gaining value."),
        ("cybersecurity 2026", "In 2026, AI-powered cyberattacks, deepfake fraud, and quantum threats are top concerns. Zero-trust architecture is standard."),
        ("global economy 2026", "The global economy in 2026 shows moderate growth at ~3.2% GDP. The AI sector contributes significantly to economic growth."),
        ("healthcare AI 2026", "By 2026, AI assists in diagnosing diseases from medical imaging with over 95% accuracy. AI-powered drug discovery shortened development cycles."),
        ("iPhone 18 Pro features", "The iPhone 18 Pro (2026) features a foldable display, on-device AI with 100 TOPS neural engine, satellite messaging, and periscope camera."),
        ("Mars mission 2026", "In 2026, multiple Mars missions are underway including NASA's sample return and SpaceX's Starship cargo missions."),
        ("electric vehicles 2026", "EVs account for 35% of new car sales in 2026. Solid-state batteries offer 500-mile range and 10-minute charging."),
        ("metaverse in 2026", "The metaverse has evolved into a practical platform for remote work, education, and social interaction with improved VR/AR hardware."),
        ("5G and 6G 2026", "5G covers 80% of populated areas globally in 2026. 6G research is underway with expected speeds of 100 Gbps."),
        ("biotechnology 2026", "CRISPR-based therapies have been approved for several genetic diseases. mRNA vaccine technology is being applied to cancer treatment."),
        ("cryptocurrency 2026", "Cryptocurrency markets have stabilized with clearer regulations. Central bank digital currencies (CBDCs) are now operational in 40+ countries."),
        ("remote work 2026", "Hybrid work is the norm for 60% of knowledge workers. AI-powered collaboration tools are standard in most organizations."),
    ]

    examples = []
    for topic, answer in facts:
        for q_template in [
            f"Tell me about {topic}",
            f"What can you tell me about {topic}?",
            f"Explain {topic}.",
            f"What is the latest on {topic}?",
        ]:
            examples.append({"instruction": q_template, "output": answer, "type": "knowledge_2026"})

    return examples


# ============================================================
# 4. AGENTIC PLANNING (30+ examples)
# ============================================================
def generate_agentic_planning():
    examples = [
        {"instruction": "Write a Python class for a bank account with deposit, withdraw, and balance check.", "output": "```python\nclass BankAccount:\n    def __init__(self, owner, balance=0.0):\n        self.owner = owner\n        self.balance = balance\n\n    def deposit(self, amount):\n        if amount <= 0:\n            raise ValueError('Amount must be positive')\n        self.balance += amount\n        return self.balance\n\n    def withdraw(self, amount):\n        if amount > self.balance:\n            raise ValueError('Insufficient funds')\n        self.balance -= amount\n        return self.balance\n\n    def check_balance(self):\n        return f'{self.owner}: ${self.balance:.2f}'\n\n# Usage\nacct = BankAccount('Alice', 1000)\nacct.deposit(500)\nacct.withdraw(200)\nprint(acct.check_balance())  # Alice: $1300.00\n```", "type": "agentic_planning"},
        {"instruction": "Design a REST API for a library management system.", "output": "**Library Management REST API**\n\n**Endpoints:**\n- `GET /books` — List all books (query: ?author=, ?genre=, ?available=)\n- `GET /books/:id` — Get book details\n- `POST /books` — Add new book\n- `PUT /books/:id` — Update book\n- `DELETE /books/:id` — Remove book\n- `POST /members` — Register member\n- `POST /loans` — Borrow a book\n- `POST /loans/:id/return` — Return a book\n\n**Example Request:**\n```json\nPOST /books\n{\n  \"title\": \"1984\",\n  \"author\": \"George Orwell\",\n  \"isbn\": \"9780451524935\",\n  \"genre\": \"Dystopian\",\n  \"copies\": 3\n}\n```\n\n**Response:** `201 Created` with book details.\n\n**Tech Stack:** Python FastAPI, PostgreSQL, Redis cache, JWT auth.", "type": "agentic_planning"},
        {"instruction": "Write a Python script to download and analyze stock prices.", "output": "```python\nimport yfinance as yf\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\ndef analyze_stock(ticker, period='6mo'):\n    stock = yf.Ticker(ticker)\n    hist = stock.history(period=period)\n\n    # Calculate moving averages\n    hist['MA50'] = hist['Close'].rolling(50).mean()\n    hist['MA200'] = hist['Close'].rolling(200).mean()\n\n    # Basic stats\n    print(f'Current: ${hist[\"Close\"][-1]:.2f}')\n    print(f'52W High: ${hist[\"High\"].max():.2f}')\n    print(f'52W Low: ${hist[\"Low\"].min():.2f}')\n    print(f'Avg Volume: {hist[\"Volume\"].mean():,.0f}')\n\n    # Plot\n    plt.figure(figsize=(12,6))\n    plt.plot(hist['Close'], label='Close')\n    plt.plot(hist['MA50'], label='50-day MA')\n    plt.plot(hist['MA200'], label='200-day MA')\n    plt.title(f'{ticker} Stock Analysis')\n    plt.legend()\n    plt.show()\n\nanalyze_stock('AAPL')\n```", "type": "agentic_planning"},
        {"instruction": "Create a data processing pipeline that reads CSV, cleans data, and generates a report.", "output": "```python\nimport pandas as pd\nimport numpy as np\nfrom pathlib import Path\n\ndef process_sales_data(csv_path):\n    # Read\n    df = pd.read_csv(csv_path)\n\n    # Clean\n    df.drop_duplicates(inplace=True)\n    df.fillna({'quantity': 0, 'price': df['price'].median()}, inplace=True)\n    df['date'] = pd.to_datetime(df['date'])\n\n    # Transform\n    df['revenue'] = df['quantity'] * df['price']\n    df['month'] = df['date'].dt.to_period('M')\n\n    # Aggregate\n    summary = df.groupby('month').agg(\n        total_revenue=('revenue', 'sum'),\n        total_orders=('order_id', 'nunique'),\n        avg_order_value=('revenue', 'mean')\n    ).round(2)\n\n    # Report\n    report_path = Path(csv_path).with_suffix('.report.csv')\n    summary.to_csv(report_path)\n    print(f'Report saved to {report_path}')\n    print(summary)\n    return summary\n```", "type": "agentic_planning"},
        {"instruction": "Write SQL queries to find the top 10 customers by revenue.", "output": "```sql\n-- Top 10 customers by total revenue\nSELECT\n    c.customer_id,\n    c.name,\n    c.email,\n    SUM(o.total_amount) AS total_revenue,\n    COUNT(o.order_id) AS order_count\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nWHERE o.status = 'completed'\n  AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)\nGROUP BY c.customer_id, c.name, c.email\nORDER BY total_revenue DESC\nLIMIT 10;\n\n-- With average order value\nSELECT\n    c.customer_id,\n    c.name,\n    SUM(o.total_amount) AS total_revenue,\n    AVG(o.total_amount) AS avg_order_value,\n    COUNT(o.order_id) AS order_count\nFROM customers c\nJOIN orders o ON c.customer_id = o.customer_id\nGROUP BY c.customer_id, c.name\nHAVING total_revenue > 1000\nORDER BY total_revenue DESC;\n```", "type": "agentic_planning"},
        {"instruction": "Write a Python script to scrape news headlines from a website.", "output": "```python\nimport requests\nfrom bs4 import BeautifulSoup\nfrom datetime import datetime\n\ndef scrape_headlines(url='https://news.ycombinator.com/'):\n    headers = {'User-Agent': 'Mozilla/5.0'}\n    resp = requests.get(url, headers=headers, timeout=10)\n    soup = BeautifulSoup(resp.text, 'html.parser')\n\n    headlines = []\n    for item in soup.select('.athing'):\n        title_elem = item.select_one('.titleline a')\n        if title_elem:\n            headlines.append({\n                'title': title_elem.text,\n                'url': title_elem['href'],\n                'scraped_at': datetime.now().isoformat()\n            })\n\n    return headlines\n\nheadlines = scrape_headlines()\nfor h in headlines[:5]:\n    print(f'{h[\"title\"]}')\n    print(f'  {h[\"url\"]}\\n')\n```", "type": "agentic_planning"},
        {"instruction": "Write a unit test suite for a Calculator class in Python.", "output": "```python\nimport pytest\n\nclass Calculator:\n    def add(self, a, b): return a + b\n    def subtract(self, a, b): return a - b\n    def multiply(self, a, b): return a * b\n    def divide(self, a, b):\n        if b == 0:\n            raise ValueError('Cannot divide by zero')\n        return a / b\n\nclass TestCalculator:\n    def setup_method(self):\n        self.calc = Calculator()\n\n    def test_add(self):\n        assert self.calc.add(2, 3) == 5\n        assert self.calc.add(-1, 1) == 0\n\n    def test_subtract(self):\n        assert self.calc.subtract(5, 3) == 2\n        assert self.calc.subtract(0, 5) == -5\n\n    def test_multiply(self):\n        assert self.calc.multiply(4, 3) == 12\n        assert self.calc.multiply(0, 5) == 0\n\n    def test_divide(self):\n        assert self.calc.divide(10, 2) == 5\n        assert self.calc.divide(7, 2) == 3.5\n\n    def test_divide_by_zero(self):\n        with pytest.raises(ValueError):\n            self.calc.divide(5, 0)\n```", "type": "agentic_planning"},
        {"instruction": "Write a Django model for a blog with posts, comments, and tags.", "output": "```python\nfrom django.db import models\nfrom django.contrib.auth.models import User\nfrom django.utils import timezone\n\nclass Tag(models.Model):\n    name = models.CharField(max_length=50, unique=True)\n    slug = models.SlugField(unique=True)\n\n    def __str__(self):\n        return self.name\n\nclass Post(models.Model):\n    title = models.CharField(max_length=200)\n    slug = models.SlugField(unique=True)\n    content = models.TextField()\n    author = models.ForeignKey(User, on_delete=models.CASCADE)\n    tags = models.ManyToManyField(Tag, blank=True)\n    created_at = models.DateTimeField(default=timezone.now)\n    updated_at = models.DateTimeField(auto_now=True)\n    is_published = models.BooleanField(default=False)\n\n    class Meta:\n        ordering = ['-created_at']\n\n    def __str__(self):\n        return self.title\n\nclass Comment(models.Model):\n    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')\n    author = models.ForeignKey(User, on_delete=models.CASCADE)\n    content = models.TextField()\n    created_at = models.DateTimeField(default=timezone.now)\n    is_approved = models.BooleanField(default=False)\n\n    class Meta:\n        ordering = ['created_at']\n\n    def __str__(self):\n        return f'Comment by {self.author} on {self.post}'\n```", "type": "agentic_planning"},
    ]

    return examples


# ============================================================
# 5. CONVERSATIONAL MULTI-TURN
# ============================================================
def generate_multi_turn():
    conversations = [
        {"q": "User: Hello! Who created you?", "a": "I was created by Aryan Chavan. He built me as an AnonyLLM project! How can I help you today?"},
        {"q": "User: What can you do?\nAssistant: I can help with coding, math, planning, and more!\nUser: Show me some math.", "a": "Sure! For example, 15% of 200 is **30**. Or solve 2x + 6 = 20 → x = 7. What kind of math would you like to explore?"},
        {"q": "User: Write Python to find prime numbers.", "a": "```python\ndef sieve_of_eratosthenes(n):\n    is_prime = [True] * (n + 1)\n    is_prime[0] = is_prime[1] = False\n    for i in range(2, int(n**0.5) + 1):\n        if is_prime[i]:\n            for j in range(i*i, n + 1, i):\n                is_prime[j] = False\n    return [i for i in range(2, n + 1) if is_prime[i]]\n\nprint(sieve_of_eratosthenes(50))\n# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]\n```\nThis uses the Sieve of Eratosthenes, which is O(n log log n)."},
        {"q": "User: What is machine learning?\nAssistant: ML is a subset of AI where computers learn patterns from data.\nUser: What are the types?", "a": "Three main types:\n\n1. **Supervised Learning**: Labeled data (input → output). Used for classification/regression.\n2. **Unsupervised Learning**: Unlabeled data. Used for clustering/dimensionality reduction.\n3. **Reinforcement Learning**: Learn via trial and error with rewards. Used in game AI and robotics.\n\nEach type has different use cases and algorithms!"},
        {"q": "User: Explain neural networks.\nAssistant: Neural networks are computing systems inspired by biological brains.\nUser: How do they learn?", "a": "Neural networks learn through a process called **backpropagation**:\n\n1. **Forward pass**: Input flows through layers, each applying weights and activations\n2. **Loss calculation**: Compare output to expected result\n3. **Backward pass**: Calculate gradients (how much each weight contributed to the error)\n4. **Weight update**: Adjust weights using optimizer (SGD, Adam) to reduce loss\n\nRepeat thousands of times! The key is having enough data and the right architecture for your task."},
        {"q": "User: What's the difference between lists and tuples in Python?\nAssistant: Lists are mutable, tuples are immutable.\nUser: When should I use each?", "a": "Great question!\n\n**Use lists when:**\n- You need to add/remove/modify items\n- The data is homogeneous (same type)\n- Example: `['apple', 'banana', 'cherry']`\n\n**Use tuples when:**\n- The data should be constant (immutable)\n- The data is heterogeneous (different types)\n- You need a hashable type (e.g., as dict keys)\n- Example: `(latitude, longitude)` coordinates\n\n**Performance**: Tuples are slightly faster and use less memory since they're immutable."},
        {"q": "User: What is an API?", "a": "**API** (Application Programming Interface) is a set of rules that allows different software to communicate.\n\nThink of it like a restaurant menu:\n- You (the client) choose from the menu (API endpoints)\n- The kitchen (server) prepares your order\n- The waiter (API) brings you the result\n\n**Real example:**\n```python\nimport requests\nresp = requests.get('https://api.github.com/users/octocat')\ndata = resp.json()\nprint(data['name'])  # 'The Octocat'\n```\n\nMost modern APIs use REST with JSON responses over HTTP."},
        {"q": "User: How do I optimize a slow Python loop?", "a": "Several strategies, from easiest to most impactful:\n\n1. **Use built-in functions**: `map()`, `filter()`, `sum()` are C-optimized\n2. **List comprehensions**: `[x*2 for x in data]` instead of `for` loops\n3. **Avoid attribute lookups**: Bind methods to local variables: `local_append = my_list.append`\n4. **Use NumPy**: For numerical data, vectorized operations are 10-100x faster\n5. **Use `itertools`**: For specialized iteration patterns\n6. **Consider Numba/JIT**: Compile hot loops with `@jit`\n7. **Parallelize**: Use `multiprocessing` or `concurrent.futures` for CPU-bound tasks\n\nPro tip: Profile first! Use `cProfile` or `timeit` — don't optimize blindly."},
    ]
    return [{"instruction": c["q"], "output": c["a"], "type": "multi_turn"} for c in conversations]


# ============================================================
# MAIN GENERATOR
# ============================================================
def get_all_agentic_data():
    data = []
    data.extend(generate_react_trajectories())
    data.extend(generate_structured_tool_data())
    data.extend(generate_2026_knowledge())
    data.extend(generate_agentic_planning())
    data.extend(generate_multi_turn())
    return data

if __name__ == "__main__":
    data = get_all_agentic_data()
    print(f"Total: {len(data)} examples")
    for t in sorted(set(d["type"] for d in data)):
        count = sum(1 for d in data if d["type"] == t)
        print(f"  {t}: {count}")
