#!/usr/bin/env python3
"""
Sample Training Data Generator for LLM
Creates various types of training datasets in different formats
"""

import json
import random
import os
from typing import List, Dict

def generate_conversational_data() -> List[Dict[str, str]]:
    """Generate conversational training data similar to ChatGPT"""
    conversations = [
        {
            "text": "Human: What is artificial intelligence?\nAssistant: Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. There are different types of AI, including narrow AI (designed for specific tasks) and general AI (theoretical human-level intelligence across all domains)."
        },
        {
            "text": "Human: How do neural networks work?\nAssistant: Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) organized in layers. Each connection has a weight that determines its strength. The network processes input data through these layers, adjusting weights during training to minimize errors. The basic process involves: 1) Forward propagation - data flows through the network, 2) Loss calculation - comparing output to expected results, 3) Backpropagation - adjusting weights to reduce errors. This process repeats until the network learns to make accurate predictions."
        },
        {
            "text": "Human: Write a Python function to calculate fibonacci numbers.\nAssistant: Here's a Python function to calculate Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# More efficient iterative version:\ndef fibonacci_iterative(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```\n\nThe first version uses recursion but can be slow for large numbers. The iterative version is more efficient for larger values of n."
        },
        {
            "text": "Human: Explain quantum computing in simple terms.\nAssistant: Quantum computing uses the strange properties of quantum physics to process information differently than regular computers. While classical computers use bits that are either 0 or 1, quantum computers use quantum bits (qubits) that can be both 0 and 1 simultaneously - this is called superposition. Additionally, qubits can be entangled, meaning they're mysteriously connected even when separated. These properties allow quantum computers to explore many possible solutions at once, potentially solving certain problems much faster than classical computers, especially in areas like cryptography, drug discovery, and optimization."
        },
        {
            "text": "Human: What are the main components of a computer?\nAssistant: The main components of a computer include:\n\n1. **CPU (Central Processing Unit)**: The brain that executes instructions\n2. **RAM (Random Access Memory)**: Temporary storage for active programs\n3. **Storage**: Permanent data storage (hard drives, SSDs)\n4. **Motherboard**: Connects all components together\n5. **Power Supply**: Provides electricity to all components\n6. **GPU (Graphics Processing Unit)**: Handles visual processing\n7. **Input/Output devices**: Keyboard, mouse, monitor, etc.\n\nThese components work together to process, store, and display information."
        },
        {
            "text": "Human: How do I learn programming?\nAssistant: Here's a structured approach to learning programming:\n\n1. **Choose a beginner-friendly language**: Python, JavaScript, or Java are good starting points\n2. **Learn the basics**: Variables, data types, loops, conditionals, functions\n3. **Practice regularly**: Code every day, even if just for 30 minutes\n4. **Build projects**: Start small (calculator, to-do list) and gradually increase complexity\n5. **Use online resources**: Codecademy, freeCodeCamp, Khan Academy\n6. **Join communities**: Stack Overflow, Reddit programming communities\n7. **Read others' code**: Study open-source projects on GitHub\n8. **Debug systematically**: Learn to troubleshoot and fix errors\n9. **Stay consistent**: Programming skills develop through regular practice\n\nRemember, programming is like learning a language - it takes time and practice!"
        },
        {
            "text": "Human: What is machine learning?\nAssistant: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task. Instead of following pre-written instructions, ML systems identify patterns in data and make predictions or decisions based on those patterns.\n\nThere are three main types:\n1. **Supervised Learning**: Learning from labeled examples (like email spam detection)\n2. **Unsupervised Learning**: Finding hidden patterns in unlabeled data (like customer segmentation)\n3. **Reinforcement Learning**: Learning through trial and error with rewards and penalties (like game playing)\n\nCommon applications include recommendation systems, image recognition, natural language processing, and autonomous vehicles."
        },
        {
            "text": "Human: Explain the difference between AI, ML, and deep learning.\nAssistant: AI, ML, and deep learning are related but distinct concepts:\n\n**Artificial Intelligence (AI)**: The broadest term, referring to any technique that enables machines to mimic human intelligence. This includes rule-based systems, expert systems, and machine learning.\n\n**Machine Learning (ML)**: A subset of AI that focuses on algorithms that can learn from data without being explicitly programmed. ML systems improve their performance on a task through experience.\n\n**Deep Learning**: A subset of ML that uses artificial neural networks with multiple layers (hence 'deep') to model and understand complex patterns. It's inspired by the human brain's structure.\n\nThink of it as nested circles: AI contains ML, and ML contains deep learning. Deep learning has driven many recent AI breakthroughs in image recognition, natural language processing, and game playing."
        },
        {
            "text": "Human: What programming language should I learn first?\nAssistant: For beginners, I recommend starting with **Python** for these reasons:\n\n**Advantages of Python:**\n- Simple, readable syntax that's close to English\n- Large, supportive community\n- Versatile (web development, data science, AI, automation)\n- Extensive libraries and frameworks\n- High demand in job market\n\n**Other good options:**\n- **JavaScript**: Essential for web development, runs everywhere\n- **Java**: Popular in enterprise, teaches good programming fundamentals\n- **C++**: More challenging but teaches low-level concepts\n- **Scratch**: Visual programming for absolute beginners\n\n**My recommendation**: Start with Python to learn programming concepts, then learn JavaScript if you're interested in web development, or explore other languages based on your specific interests (mobile apps, game development, etc.)."
        },
        {
            "text": "Human: How does the internet work?\nAssistant: The internet works through a global network of interconnected computers that communicate using standardized protocols. Here's a simplified explanation:\n\n1. **Physical Infrastructure**: Cables, fiber optics, wireless signals, and satellites connect devices worldwide\n\n2. **Internet Service Providers (ISPs)**: Companies that provide internet access to homes and businesses\n\n3. **Protocols**: Standard rules for communication, mainly TCP/IP (Transmission Control Protocol/Internet Protocol)\n\n4. **Domain Name System (DNS)**: Translates human-readable website names (google.com) into IP addresses that computers understand\n\n5. **Data Packets**: Information is broken into small packets, sent through various routes, and reassembled at the destination\n\n6. **Servers and Clients**: Servers host websites and data, while clients (your devices) request and receive this information\n\nWhen you visit a website, your device sends a request through your ISP to the appropriate server, which sends back the webpage data through the same network infrastructure."
        }
    ]
    return conversations

def generate_educational_content() -> List[Dict[str, str]]:
    """Generate educational content for training"""
    content = [
        {
            "text": "The scientific method is a systematic approach to understanding the natural world. It involves several key steps: observation, hypothesis formation, experimentation, data analysis, and conclusion. Scientists use this method to test ideas and build knowledge incrementally. The process is iterative, meaning that results often lead to new questions and further research. Peer review and reproducibility are crucial components that ensure scientific findings are reliable and valid."
        },
        {
            "text": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs in chloroplasts, specifically in the chlorophyll-containing structures called thylakoids. The process has two main stages: the light-dependent reactions (which capture energy from sunlight) and the light-independent reactions or Calvin cycle (which uses that energy to produce glucose). Photosynthesis is fundamental to life on Earth as it produces the oxygen we breathe and forms the base of most food chains."
        },
        {
            "text": "The water cycle describes the continuous movement of water on, above, and below Earth's surface. It includes evaporation (water turning to vapor), condensation (vapor forming clouds), precipitation (rain, snow, sleet), and collection (water gathering in bodies of water). This cycle is driven by solar energy and gravity. It's essential for distributing fresh water around the planet and supporting all life forms. Human activities can affect the water cycle through pollution, deforestation, and climate change."
        },
        {
            "text": "Democracy is a form of government where power is vested in the people, who exercise it directly or through elected representatives. Key principles include majority rule with minority rights, individual liberty, equality before the law, and regular free elections. There are different types of democracy, including direct democracy (citizens vote on issues directly) and representative democracy (citizens elect representatives to make decisions). Democratic institutions include separation of powers, checks and balances, and protection of civil liberties."
        },
        {
            "text": "The theory of evolution explains how species change over time through natural selection. Charles Darwin proposed that organisms with favorable traits are more likely to survive and reproduce, passing these traits to their offspring. Over many generations, this leads to changes in the characteristics of populations. Evidence for evolution comes from fossils, comparative anatomy, molecular biology, and direct observation of evolutionary changes in organisms like bacteria and insects. Evolution is the unifying theory of biology, explaining the diversity of life on Earth."
        }
    ]
    return content

def generate_creative_writing() -> List[Dict[str, str]]:
    """Generate creative writing samples"""
    stories = [
        {
            "text": "The old lighthouse stood sentinel against the crashing waves, its beacon cutting through the fog like a sword of light. Sarah climbed the spiral staircase, each step echoing in the hollow tower. She had inherited this place from her grandmother, along with its secrets and stories. At the top, she found the journal her grandmother had mentioned—filled with accounts of ships saved and souls lost to the merciless sea. As she read by candlelight, Sarah realized she was now the keeper of more than just a lighthouse; she was the guardian of maritime memories."
        },
        {
            "text": "In the year 2157, Maya discovered an ancient seed buried beneath the ruins of New Tokyo. The world had been barren for decades, but this seed pulsed with an otherworldly energy. Against all regulations, she planted it in her rooftop garden. Within days, it sprouted into something magnificent—a tree that seemed to sing with the wind and whose leaves shimmered like stars. As word spread, people came from across the wasteland to see the miracle. The tree became a symbol of hope, proving that even in the darkest times, life finds a way to flourish."
        },
        {
            "text": "The clockmaker's shop existed between seconds, in the spaces where time hiccupped and reality caught its breath. Elias had stumbled upon it during a thunderstorm, seeking shelter from the rain. Inside, hundreds of clocks ticked in perfect discord, each keeping time for a different world. The proprietor, an ancient woman with silver hair, offered him tea that tasted like childhood summers. 'Time,' she said, 'is not a river flowing in one direction. It's an ocean, and some of us know how to swim.' When Elias emerged, only moments had passed in his world, but he carried memories of lifetimes."
        },
        {
            "text": "The dragon's library was renowned throughout the realm, not for its treasure, but for its books. Centuries of collecting had filled every cavern with volumes from across the world. When young Thomas arrived seeking the cure for his village's plague, he expected to battle the beast. Instead, he found Seraphina, the dragon, hunched over a medical tome, spectacles perched on her snout. 'Ah,' she said without looking up, 'another seeker of knowledge. The cure you need is in aisle seven, third shelf from the top. But first, would you like some tea and a discussion about medieval herbalism?'"
        }
    ]
    return stories

def generate_technical_content() -> List[Dict[str, str]]:
    """Generate technical documentation and coding examples"""
    technical = [
        {
            "text": "Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which contain data (attributes) and code (methods). The four main principles of OOP are:\n\n1. Encapsulation: Bundling data and methods that work on that data within a single unit\n2. Inheritance: Creating new classes based on existing classes\n3. Polymorphism: Using a single interface to represent different types\n4. Abstraction: Hiding complex implementation details\n\nExample in Python:\n```python\nclass Animal:\n    def __init__(self, name):\n        self.name = name\n    \n    def speak(self):\n        pass\n\nclass Dog(Animal):\n    def speak(self):\n        return f'{self.name} says Woof!'\n```"
        },
        {
            "text": "Database normalization is the process of organizing data in a database to reduce redundancy and improve data integrity. The main normal forms are:\n\n1. First Normal Form (1NF): Each column contains atomic values, no repeating groups\n2. Second Normal Form (2NF): Meets 1NF and all non-key attributes depend on the entire primary key\n3. Third Normal Form (3NF): Meets 2NF and no transitive dependencies exist\n\nNormalization helps prevent data anomalies during insert, update, and delete operations. However, sometimes denormalization is used for performance optimization in data warehouses and analytical systems."
        },
        {
            "text": "REST (Representational State Transfer) is an architectural style for designing web services. RESTful APIs use HTTP methods to perform operations on resources identified by URLs. The main principles include:\n\n- Stateless: Each request contains all necessary information\n- Client-server architecture: Separation of concerns\n- Cacheable: Responses can be cached for performance\n- Uniform interface: Consistent way to interact with resources\n- Layered system: Hierarchical layers for scalability\n\nHTTP methods in REST:\n- GET: Retrieve data\n- POST: Create new resource\n- PUT: Update existing resource\n- DELETE: Remove resource\n- PATCH: Partial update"
        },
        {
            "text": "Version control systems like Git help developers track changes in code over time. Key concepts include:\n\n- Repository: Storage location for project files and history\n- Commit: Snapshot of changes with a descriptive message\n- Branch: Parallel development line for features or experiments\n- Merge: Combining changes from different branches\n- Remote: Repository stored on a server (like GitHub)\n\nBasic Git workflow:\n1. git clone: Copy repository locally\n2. git add: Stage changes\n3. git commit: Save changes with message\n4. git push: Upload changes to remote\n5. git pull: Download latest changes\n\nBranching allows multiple developers to work on different features simultaneously without conflicts."
        }
    ]
    return technical

def generate_qa_pairs() -> List[Dict[str, str]]:
    """Generate question-answer pairs for training"""
    qa_pairs = [
        {
            "text": "Q: What is the capital of France?\nA: The capital of France is Paris. It's located in the north-central part of the country and is known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."
        },
        {
            "text": "Q: How many planets are in our solar system?\nA: There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006."
        },
        {
            "text": "Q: What is the largest ocean on Earth?\nA: The Pacific Ocean is the largest ocean on Earth, covering about one-third of the planet's surface. It's larger than all the land masses combined."
        },
        {
            "text": "Q: Who wrote 'Romeo and Juliet'?\nA: William Shakespeare wrote 'Romeo and Juliet' around 1594-1596. It's one of his most famous tragedies and tells the story of two young star-crossed lovers."
        },
        {
            "text": "Q: What is the speed of light?\nA: The speed of light in a vacuum is approximately 299,792,458 meters per second (about 186,282 miles per second). This is a fundamental constant in physics, denoted by 'c'."
        }
    ]
    return qa_pairs

def create_training_files():
    """Create various training data files"""
    
    # Create output directory
    os.makedirs("training_data", exist_ok=True)
    
    # 1. Conversational data (JSON format)
    conversations = generate_conversational_data()
    with open("training_data/conversations.json", "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    # 2. Mixed content (JSON format)
    all_content = []
    all_content.extend(generate_educational_content())
    all_content.extend(generate_creative_writing())
    all_content.extend(generate_technical_content())
    all_content.extend(generate_qa_pairs())
    
    with open("training_data/mixed_content.json", "w", encoding="utf-8") as f:
        json.dump(all_content, f, indent=2, ensure_ascii=False)
    
    # 3. Plain text format (all content combined)
    with open("training_data/combined_text.txt", "w", encoding="utf-8") as f:
        for item in all_content + conversations:
            f.write(item["text"] + "\n\n")
    
    # 4. Large synthetic dataset
    large_dataset = []
    
    # Generate more conversational data
    topics = [
        "climate change", "space exploration", "renewable energy", "blockchain technology",
        "virtual reality", "genetic engineering", "quantum physics", "sustainable agriculture",
        "urban planning", "mental health", "education technology", "autonomous vehicles"
    ]
    
    for topic in topics:
        large_dataset.append({
            "text": f"Human: Can you explain {topic}?\nAssistant: {topic.title()} is a complex and fascinating subject that involves multiple aspects and considerations. It requires understanding of scientific principles, technological capabilities, social implications, and economic factors. The field has evolved significantly over recent years with new developments and research findings that continue to shape our understanding and approach to this area."
        })
    
    # Add programming examples
    programming_languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust", "Ruby", "PHP"]
    for lang in programming_languages:
        large_dataset.append({
            "text": f"Human: What are the key features of {lang}?\nAssistant: {lang} is a programming language with several distinctive features that make it suitable for various types of software development. Its design philosophy, syntax characteristics, performance considerations, and ecosystem of libraries and frameworks contribute to its popularity and effectiveness in different domains of programming and software engineering."
        })
    
    with open("training_data/large_dataset.json", "w", encoding="utf-8") as f:
        json.dump(large_dataset, f, indent=2, ensure_ascii=False)
    
    # 5. Instruction-following format
    instruction_data = [
        {
            "text": "Instruction: Write a function to reverse a string.\nInput: hello world\nOutput: dlrow olleh\n\nHere's a Python function to reverse a string:\n```python\ndef reverse_string(s):\n    return s[::-1]\n```"
        },
        {
            "text": "Instruction: Explain the concept in simple terms.\nInput: Photosynthesis\nOutput: Photosynthesis is how plants make food using sunlight, water, and air. They take in carbon dioxide from the air and water from their roots, then use sunlight to turn these into sugar for energy and release oxygen that we breathe."
        },
        {
            "text": "Instruction: Summarize the following text.\nInput: Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, scientific evidence shows that human activities have been the main driver since the 1800s.\nOutput: Climate change involves long-term changes in global weather patterns, primarily caused by human activities since the 1800s."
        }
    ]
    
    with open("training_data/instruction_following.json", "w", encoding="utf-8") as f:
        json.dump(instruction_data, f, indent=2, ensure_ascii=False)
    
    print("Training data files created successfully!")
    print("\nGenerated files:")
    print("- training_data/conversations.json (conversational AI data)")
    print("- training_data/mixed_content.json (educational, creative, technical content)")
    print("- training_data/combined_text.txt (all content in plain text)")
    print("- training_data/large_dataset.json (expanded synthetic dataset)")
    print("- training_data/instruction_following.json (instruction-response pairs)")
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"- Conversations: {len(conversations)} examples")
    print(f"- Mixed content: {len(all_content)} examples")
    print(f"- Large dataset: {len(large_dataset)} examples")
    print(f"- Instruction data: {len(instruction_data)} examples")
    
    # Calculate approximate token counts (rough estimate)
    total_chars = sum(len(item["text"]) for item in all_content + conversations + large_dataset + instruction_data)
    estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
    print(f"- Estimated total tokens: {estimated_tokens:,}")

if __name__ == "__main__":
    create_training_files()