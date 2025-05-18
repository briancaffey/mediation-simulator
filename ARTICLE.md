## AI in the (Mock) Courtroom: Building Mediation Simulator for the NVIDIA Hackathon!

I'm excited to share a project I've been building for the NVIDIA Agent Intelligence Toolkit Hackathon: **Mediation Simulator**! The NVIDIA Agent Intelligence Toolkit (or AgentIQ Toolkit, as it's often called) is a powerful open-source library designed for connecting, evaluating, and accelerating teams of AI agents. My goal? To see if I could leverage this toolkit to build AI agent teams capable of simulating the entire, complex process of a law school mediation competition.

### What is Mediation Simulator? And Why Mediation?

At its core, Mediation Simulator is my attempt to model the nuanced, semi-structured, three-way conversations that happen in mediation. Law school mediation tournaments are fascinating events where students practice crucial negotiation and dispute resolution skills. This seemed like a really interesting challenge! How do you get language models to effectively navigate such a dynamic environment?

Mediation competitions also present some unique data challenges that are perfect for AI:
*   **Layered Information:** There are "common facts" known to everyone, and "confidential facts" privy only to one party (and sometimes shared strategically with the mediator or the other side).
*   **Dynamic Conversations:** The main discussion involves all three parties (mediator and two disputants), but it also features "caucuses"—private, two-way conversations between the mediator and one party.
*   **Creative Scenarios:** The cases are entirely fictional, often involving made-up companies, fictional countries, and even fictional currencies! This is a deliberate choice to help students avoid real-world biases. And guess what? AI is really good at generating fake data!

### The Genesis: From Idea to Data

The first major hurdle was generating the foundational case data. Here’s how that unfolded:

1.  **Deep Dive Research:** I started by brainstorming with an LLM (Opus, in this case) to get a comprehensive understanding of law school mediation competitions. I wanted to know everything: the rules, structure, participants, judging criteria, types of cases—the works!
2.  **Prompt Engineering for Cases:** With that knowledge, I tasked another LLM (GPT-4o) with generating a robust prompt that could, in turn, be used to create diverse and realistic (albeit fictional) mediation case scenarios.
3.  **Case Generation:** Using this master prompt, I generated a set of distinct cases.
4.  **Structuring with LangGraph:** To manage the data for each case, I turned to LangGraph. I designed a state object to encapsulate all crucial elements:
    *   Common facts.
    *   Confidential facts for both the requesting and responding parties.
    *   **Related Documents!** This was my own little twist. I wanted to test RAG (Retrieval Augmented Generation) integration within the agentic workflow. Could parties use tools to search for information in these documents to bolster their arguments during mediation?
5.  **Data Persistence:** With the data structured, I saved it in accessible formats: the LangGraph state was serialized to YAML, and the case details (like facts and documents) were stored in Markdown files.

### Orchestrating the Simulation: An Agentic Ballet

With the case data ready, the next step was to architect the mediation simulation itself:

1.  **Defining the Flow:** I broke down the mediation process into its typical phases:
    *   Opening Statements
    *   Information Gathering
    *   Caucuses (for each party)
    *   Negotiation
    *   Conclusion
2.  **Assembling the Agent Team:** I set up a LangGraph graph consisting of:
    *   A **Mediator** agent.
    *   Two **Party** agents (Requesting and Responding).
    *   A **Clerk** agent, whose job is to help manage the conversation flow and transition the simulation between the different phases.
3.  **Dynamic Prompting:** The prompts for the mediator and the parties (both when initiating a statement and when responding) change significantly based on the current phase of the mediation. Critically, these prompts also dynamically include a summary log of what has already been said, providing context.
4.  **Event Logging:** Each time a party speaks, I store that interaction in an event log. This log tracks who said what, during which phase, and includes a brief description of their statement, modeling the flow of the discussion.

Getting to a functional MVP (Minimum Viable Product) was crucial! It allowed me to see the actual dialogue unfold and immediately highlighted areas for improvement. For instance, I realized that prompts needed to guide parties to ask one clear question at a time, directed at a specific participant, rather than posing multiple questions to several people at once. This really helps keep the simulated conversation straightforward and more realistic.

### Bringing it to Life: The "Vibe Coding" Web Viewer!

Reading through raw Markdown files or terminal output to follow a complex, multi-turn mediation isn't ideal. I needed a better way to visualize the results! And this is where a bit of "vibe coding" came in incredibly handy.

I prompted an LLM to generate a single HTML page using Vue.js and Tailwind CSS. My requirements were simple: list all generated mediation cases, and when a case is selected, display the full dialogue. The amazing part? I never actually looked at the generated code in detail! I was able to make incremental improvements by simply describing changes or new features I wanted, and the LLM iterated until it was almost exactly what I envisioned. This was super easy and fast!

Having this simple web interface has been a game-changer for reviewing simulations. Plus, keeping it in my project repository means I can easily host it on GitHub Pages at `briancaffey.github.io/mediation-simulator` (once it's ready!). It’s just so much better than staring at text files!

### My Take on the NVIDIA AgentIQ Toolkit

Now, I know what some developers think about LLM frameworks like LangChain/LangGraph, LlamaIndex, and CrewAI—there's often a bit of "framework fatigue." But hear me out!

The AgentIQ Toolkit is doing something really valuable by bringing patterns and components from these (and other) frameworks into a cohesive system. This allows for some truly interesting and powerful combinations. The examples provided are excellent and taught me a lot about new patterns for building agentic workflows. ReWOO agents, for instance, were a new concept for me, as was seeing how to effectively combine LangGraph with LlamaIndex.

What I particularly appreciated was:
*   **Standardized Patterns:** AgentIQ promotes good practices for crucial aspects of AI development, like evaluations and telemetry/tracing.
*   **YAML Configuration:** I really like using YAML for configuring development environments, similar to Docker Compose. It standardizes things and vastly improves readability. The way AgentIQ allows registering functions that can be included in workflows via YAML config files is a great example of this.
*   **A Learning Goldmine:** Seriously, working through as many examples as possible was incredibly beneficial. Reading the code helped me grasp the patterns underpinning AgentIQ. You are pretty much guaranteed to learn something new!

I’m so glad I took the plunge and got my feet wet with the AgentIQ Toolkit. It's been a fantastic learning resource.

### Mediation Competitions, But for LLMs?

One of the fun, forward-looking ideas this project sparks is the concept of "mediation competitions, but for LLMs." Imagine pitting different LLMs against each other, representing the requesting and responding parties, to see how they fare in these complex negotiation scenarios!

### What's Next?

This hackathon project has been an incredible learning journey. Building Mediation Simulator has not only been a fun technical challenge but has also opened my eyes to the potential of AI agents in simulating complex human interactions. I'm excited to continue refining it and exploring more possibilities with the AgentIQ Toolkit!
