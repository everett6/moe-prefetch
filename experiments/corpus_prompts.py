"""
The capture corpus: what the predictor gets to see.

m2_capture.py used 40 hand-written prompts across 8 registers. That was enough to
establish a register-level split, and too small for anything else: 3,625 token
positions, one prompt per topic, and every register covered by five prompts whose
subject matter barely overlapped the ones held out.

Expert routing is a function of the hidden state and the hidden state is a
function of subject matter, so corpus breadth is not decoration here -- a probe
fitted on five prose prompts and tested on five more is being asked a much easier
question than the deployed predictor faces.

This builds a larger corpus in two parts, and keeps them labelled:

  seed        the original 40, verbatim, so the earlier result stays reproducible
  generated   template x topic combinations, which give genuine variation in
              subject matter while keeping the source readable

Templates alone would make a register internally homogeneous, which is why the
topic lists are long and concrete rather than abstract. The composition is
recorded in the dataset manifest, so anyone reading a number from this corpus can
see what it was measured on.

Registers are the unit of the train/val/test split, and prompts are the unit
inside a register -- so a held-out register is genuinely unseen subject matter,
not a paraphrase of something in training.
"""

SEED_PROMPTS = [
    ("code", "Write a Python function that reverses a singly linked list."),
    ("code", "Implement binary search over a sorted list, with docstring."),
    ("code", "Write a class for a least-recently-used cache with get and put."),
    ("code", "Fix this bug: def add(a, b): return a - b"),
    ("code", "Write a SQL query joining orders and customers by id."),
    ("math", "What is 156 divided by 12?"),
    ("math", "A train travels 240 km in 3 hours. What is its average speed?"),
    ("math", "Compute the derivative of x^3 + 2x^2 - 5x + 1."),
    ("math", "If 5 machines take 5 minutes to make 5 widgets, how long for 100?"),
    ("math", "Explain why the square root of 2 is irrational."),
    ("science", "Explain why the sky appears blue."),
    ("science", "How do vaccines train the immune system?"),
    ("science", "Describe the carbon cycle in a few paragraphs."),
    ("science", "What causes the seasons on Earth?"),
    ("science", "Explain how a refrigerator moves heat against a gradient."),
    ("history", "Summarize the causes of the French Revolution."),
    ("history", "What led to the fall of the Western Roman Empire?"),
    ("history", "Describe the significance of the printing press."),
    ("history", "What were the main outcomes of the Congress of Vienna?"),
    ("history", "Explain the origins of the Silk Road."),
    ("creative", "Write a short poem about winter."),
    ("creative", "Write a story about a robot discovering music."),
    ("creative", "Write a fable about a fox and a river."),
    ("creative", "Describe a lighthouse at dawn, in vivid prose."),
    ("creative", "Write a dialogue between two old friends reuniting."),
    ("factual", "What is the capital of France?"),
    ("factual", "Who wrote Pride and Prejudice?"),
    ("factual", "What year did the Berlin Wall fall?"),
    ("factual", "How many continents are there?"),
    ("factual", "What is the largest ocean on Earth?"),
    ("technical", "Describe how a hash map handles collisions."),
    ("technical", "Explain the CAP theorem in two sentences."),
    ("technical", "What is the difference between TCP and UDP?"),
    ("technical", "Explain how DNS resolves a domain name."),
    ("technical", "What does a load balancer do, and why?"),
    ("reasoning", "Should a small team prefer a monolith or microservices? Why?"),
    ("reasoning", "Compare electric and hydrogen cars for a cold climate."),
    ("reasoning", "Argue both sides of remote versus office work."),
    ("reasoning", "How would you decide whether to rewrite or refactor a system?"),
    ("reasoning", "What are the trade-offs of renting versus buying a home?"),
]

# register -> (templates, topics). Every template takes one topic.
GENERATORS = {
    "code": (
        ["Write a Python function that {}.",
         "Implement {} in Python, with a docstring and type hints.",
         "Write a unit test for code that {}.",
         "Refactor a function that {} to be clearer and faster.",
         "Explain, with code, how to {}.",
         "Write a command-line tool that {}."],
        ["parses an ISO 8601 timestamp into a datetime",
         "merges two sorted iterators lazily",
         "computes a rolling median over a stream",
         "validates a credit-card number with the Luhn algorithm",
         "flattens an arbitrarily nested list",
         "implements a trie with prefix search",
         "debounces repeated calls to a callback",
         "converts a CSV file to newline-delimited JSON",
         "finds the longest common subsequence of two strings",
         "implements a fixed-size ring buffer",
         "retries a network call with exponential backoff",
         "computes a topological ordering of a dependency graph"],
    ),
    "math": (
        ["Solve this and show your working: {}.",
         "Explain the intuition behind {}.",
         "Prove or disprove: {}.",
         "Work through {} step by step.",
         "Where do students usually go wrong with {}?"],
        ["the integral of x times e to the x",
         "why the harmonic series diverges",
         "the pigeonhole principle applied to birthdays",
         "solving a system of three linear equations by elimination",
         "the expected number of rolls to see every face of a die",
         "why matrix multiplication is not commutative",
         "Bayes' theorem applied to a medical test with 1% prevalence",
         "the difference between correlation and covariance",
         "finding the eigenvalues of a two by two matrix",
         "the binomial theorem for a cubic expansion",
         "why 0.999 repeating equals one",
         "computing compound interest over thirty years"],
    ),
    "science": (
        ["Explain {} to a curious adult.",
         "What is the current scientific understanding of {}?",
         "Describe the mechanism behind {}.",
         "What experiment would you design to study {}?",
         "What common misconception surrounds {}?"],
        ["how mRNA vaccines differ from live-attenuated ones",
         "why ice floats on water",
         "how CRISPR edits a genome",
         "the greenhouse effect and its feedback loops",
         "how neurons transmit signals across a synapse",
         "why antibiotics do not work on viruses",
         "how photosynthesis converts light into sugar",
         "plate tectonics and the formation of mountain ranges",
         "how the immune system distinguishes self from non-self",
         "why the speed of light is a universal limit",
         "how enzymes lower activation energy",
         "the role of mitochondria beyond energy production"],
    ),
    "history": (
        ["Summarise {} and its consequences.",
         "What were the underlying causes of {}?",
         "How did contemporaries understand {}?",
         "Describe the long-term legacy of {}.",
         "What changed, and what stayed the same, after {}?"],
        ["the Meiji Restoration",
         "the partition of India in 1947",
         "the Columbian Exchange",
         "the Black Death in fourteenth-century Europe",
         "the abolition of the transatlantic slave trade",
         "the Russian Revolution of 1917",
         "the construction of the transcontinental railroad",
         "the fall of Constantinople in 1453",
         "the Marshall Plan",
         "the Scramble for Africa",
         "the invention of double-entry bookkeeping",
         "the Chinese Cultural Revolution"],
    ),
    "creative": (
        ["Write a short scene in which {}.",
         "Write a poem about {}.",
         "Describe, in vivid prose, {}.",
         "Write the opening paragraph of a novel where {}.",
         "Write a piece of flash fiction about {}."],
        ["a cartographer maps a city that keeps changing",
         "two strangers share an umbrella in a downpour",
         "a clockmaker refuses to repair one particular watch",
         "the last bookshop in a town closes",
         "a gardener discovers a plant that grows backwards",
         "a lighthouse keeper receives a letter with no sender",
         "a translator finds a word with no equivalent",
         "a musician loses their hearing and keeps composing",
         "a child trades a marble for a secret",
         "an old train station reopens for one night",
         "a baker works through a power cut",
         "a diver finds a sunken orchard"],
    ),
    "factual": (
        ["What is {}?",
         "Give a short factual answer: {}.",
         "List the key facts about {}.",
         "When and where did {} happen?",
         "Who is associated with {}, and why?"],
        ["the highest waterfall in the world",
         "the chemical symbol for tungsten",
         "the longest river in South America",
         "the inventor of the telephone",
         "the population of Tokyo",
         "the deepest point in the ocean",
         "the first person to reach the South Pole",
         "the speed of sound at sea level",
         "the smallest country by land area",
         "the author of One Hundred Years of Solitude",
         "the year the euro entered circulation",
         "the largest desert on Earth"],
    ),
    "technical": (
        ["Explain how {} works.",
         "What are the trade-offs of {}?",
         "Describe a failure mode of {}.",
         "How would you debug a problem in {}?",
         "Compare two common approaches to {}."],
        ["TLS certificate validation",
         "consistent hashing in a distributed cache",
         "write-ahead logging in a database",
         "garbage collection in a generational heap",
         "the TCP congestion-control algorithm",
         "container networking and namespaces",
         "OAuth 2.0 authorization code flow",
         "CPU branch prediction",
         "a copy-on-write filesystem snapshot",
         "eventual consistency in a replicated store",
         "JIT compilation in a managed runtime",
         "rate limiting with a token bucket"],
    ),
    "reasoning": (
        ["Weigh the arguments: {}.",
         "What would change your mind about {}?",
         "Reason carefully about {}.",
         "What is the strongest counterargument to {}?",
         "How would you decide {}?"],
        ["whether a startup should raise venture capital or bootstrap",
         "whether to migrate a service to a new language",
         "whether nuclear power belongs in a renewable grid",
         "whether to buy insurance for a low-probability loss",
         "whether remote hiring widens or narrows opportunity",
         "whether to optimise for latency or throughput",
         "whether a city should build rail or bus rapid transit",
         "whether to open-source an internal tool",
         "whether to run a four-day working week",
         "whether standardised testing measures learning",
         "whether to rewrite a legacy system incrementally",
         "whether congestion pricing is fair"],
    ),
    "medical": (
        ["Explain {} in plain language for a patient.",
         "What is the evidence base for {}?",
         "Describe how clinicians assess {}.",
         "What are the risks and benefits of {}?"],
        ["managing type 2 diabetes with diet and exercise",
         "the difference between a virus and a bacterial infection",
         "how blood pressure medication works",
         "physiotherapy after a knee replacement",
         "why antibiotic courses should be completed",
         "how a vaccine schedule is designed for children",
         "the role of sleep in cardiovascular health",
         "screening for colorectal cancer",
         "how anaesthesia is monitored during surgery",
         "the difference between migraine and tension headache",
         "iron deficiency and its common causes",
         "rehabilitation after a stroke"],
    ),
    "legal": (
        ["Explain the concept of {} without giving legal advice.",
         "How do courts generally approach {}?",
         "What is the purpose of {} in law?",
         "Describe how {} differs between jurisdictions."],
        ["consideration in contract formation",
         "the burden of proof in civil versus criminal cases",
         "fair use in copyright",
         "limited liability for shareholders",
         "the doctrine of precedent",
         "trademark dilution",
         "employment at will",
         "the exclusionary rule for evidence",
         "data protection and lawful basis for processing",
         "adverse possession of land",
         "the difference between mediation and arbitration",
         "patent novelty and obviousness"],
    ),
    "business": (
        ["How should a company think about {}?",
         "What metrics matter for {}?",
         "Describe a common mistake in {}.",
         "Write a short memo about {}."],
        ["pricing a subscription product",
         "deciding when to hire a first salesperson",
         "managing inventory for a seasonal business",
         "structuring an employee equity plan",
         "entering a market with an entrenched incumbent",
         "reducing customer churn in a B2B product",
         "building a financial model for a new store",
         "negotiating terms with a single large supplier",
         "deciding whether to franchise",
         "measuring the return on a marketing campaign",
         "planning a phased product launch",
         "handling a supply-chain disruption"],
    ),
    "cooking": (
        ["Write a recipe for {}.",
         "Explain the technique behind {}.",
         "What goes wrong when making {}, and why?",
         "Adapt {} for a small kitchen."],
        ["a slow-braised beef shin with gremolata",
         "sourdough bread with a stiff starter",
         "a clear consomme",
         "handmade pasta without a machine",
         "a stable hollandaise",
         "kimchi fermented at room temperature",
         "a dark roux for gumbo",
         "tempered chocolate for coating",
         "a laminated croissant dough",
         "risotto without constant stirring",
         "cold-smoked salmon",
         "a custard that will not curdle"],
    ),
    "travel": (
        ["Plan a week in {}.",
         "What should a first-time visitor know about {}?",
         "Describe the character of {} in a few paragraphs.",
         "How would you travel through {} on a small budget?"],
        ["the Scottish Highlands in autumn",
         "coastal Portugal by train",
         "northern Japan in deep winter",
         "the Atacama desert",
         "the Danube from Vienna to Budapest",
         "rural Kerala",
         "the Norwegian fjords by ferry",
         "Patagonia on foot",
         "Sicily by road",
         "the Trans-Siberian route",
         "Iceland's ring road",
         "the Baltic capitals"],
    ),
    "music": (
        ["Explain {} to a beginner musician.",
         "Describe the history of {}.",
         "How would you practise {}?",
         "Analyse what makes {} distinctive."],
        ["modal interchange in pop songwriting",
         "the development of the blues scale",
         "counterpoint in Bach's inventions",
         "polyrhythm in West African drumming",
         "the role of the rhythm section in bebop",
         "tuning systems and equal temperament",
         "sonata form in the classical period",
         "synthesiser subtractive synthesis",
         "vocal harmony arrangement for three parts",
         "the evolution of the electric guitar solo",
         "microtonality outside Western music",
         "song structure in modern electronic music"],
    ),
    "philosophy": (
        ["Set out the main positions on {}.",
         "What is at stake in the debate about {}?",
         "Explain {} and one serious objection to it.",
         "How has thinking about {} changed over time?"],
        ["personal identity over time",
         "the is-ought distinction",
         "free will and determinism",
         "the trolley problem and its variants",
         "scientific realism",
         "the problem of other minds",
         "virtue ethics versus consequentialism",
         "the social contract",
         "the hard problem of consciousness",
         "moral luck",
         "the paradox of tolerance",
         "what makes an explanation good"],
    ),
    "environment": (
        ["Explain the science and policy around {}.",
         "What would meaningful progress on {} look like?",
         "Describe the trade-offs involved in {}.",
         "Who bears the cost of {}, and who benefits?"],
        ["restoring a degraded peat bog",
         "offshore wind siting and marine habitats",
         "urban heat islands and tree canopy",
         "agricultural runoff and river eutrophication",
         "recycling versus reducing plastic production",
         "reintroducing a keystone predator",
         "carbon pricing across borders",
         "water allocation in a drought-prone basin",
         "electrifying freight transport",
         "protecting pollinators in farmland",
         "managed retreat from eroding coastline",
         "lithium extraction and local water tables"],
    ),
}


def build_corpus(per_register=30):
    """Deterministic: same corpus every run, so a capture is reproducible."""
    prompts, seen = [], set()
    for reg, text in SEED_PROMPTS:
        if text not in seen:
            seen.add(text)
            prompts.append((reg, text, "seed"))
    for reg, (templates, topics) in GENERATORS.items():
        made = 0
        # Enumerate every template x topic pair exactly once: template cycles
        # fast, topic slowly. An earlier version strided both with a multiplier,
        # which looked more varied and in fact collided -- 5 distinct prompts per
        # register instead of 40, silently, because duplicates were dropped.
        n_t = len(templates)
        for i in range(n_t * len(topics)):
            if made >= per_register:
                break
            t = templates[i % n_t]
            top = topics[(i // n_t) % len(topics)]
            text = t.format(top)
            if text in seen:
                continue
            seen.add(text)
            prompts.append((reg, text, "generated"))
            made += 1
    return prompts


if __name__ == "__main__":
    import collections
    c = build_corpus()
    by = collections.Counter(r for r, _, _ in c)
    print(f"{len(c)} unique prompts across {len(by)} registers")
    for r, n in sorted(by.items()):
        print(f"  {r:<12} {n}")
