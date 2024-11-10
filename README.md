import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import networkx as nx
import spacy
from transformers import pipeline
import requests  # Added for API calls
import json
import logging
from queue import PriorityQueue
import unittest
from PIL import Image
import speech_recognition as sr
import pyttsx3
import schedule
import threading
import openai
from bs4 import BeautifulSoup
import dash
from dash import html, dcc
import plotly.graph_objs as go

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Load SpaCy English model for Natural Language Processing
nlp = spacy.load("en_core_web_sm")

class AGI:
    def __init__(self):
        # Memory Systems
        self.timeline = []
        self.feedback = []
        self.emotional_state = None
        self.goal_memory = []  # Stores goal-oriented actions
        self.short_term_memory = deque(maxlen=10)  # Increased capacity
        self.long_term_memory = []
        self.episodic_memory = []  # Stores specific experiences
        self.working_memory = []  # Handles ongoing tasks
        self.semantic_memory = {}  # For storing general knowledge
        self.procedural_memory = {}  # For storing procedures and skills

        # Knowledge Graph for Advanced Reasoning
        self.knowledge_graph = nx.Graph()

        # Planning Module
        self.current_plan = []

        # Natural Language Understanding
        self.nlp_model = nlp

        # Reinforcement Learning Parameters
        self.q_table = {}  # Simplistic Q-table for decision-making
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate

        # Performance Tracking for Learning Rate Adjustment
        self.performance_history = []

        # CNN Model
        self.cnn_model = self.build_cnn_model()  # Initialize CNN model

        # Transformer-Based NLU
        self.setup_transformer_nlu()

        # Bayesian Reasoning Parameters
        self.beliefs = {}  # Stores belief probabilities

        # Friend Recognition
        self.friends = set()  # Stores unique identifiers of friends

        # Common Sense Reasoning Parameters
        self.common_sense_cache = {}  # Simple cache for common sense relations

        # Task Scheduling
        self.task_queue = PriorityQueue()
        self.setup_scheduler()

        # Speech Recognition and Synthesis
        self.setup_speech_engine()

        # OpenAI API Setup (Ensure to set your API key)
        # self.setup_openai("your-openai-api-key")

        # Dashboard Setup
        # self.setup_dashboard()

    # ---------------------- Memory Management ----------------------

    def store_goal(self, goal_description):
        """Store a user-specified goal in memory."""
        goal = {
            "description": goal_description,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
        self.goal_memory.append(goal)
        print(f"Goal stored: {goal_description}")

    def review_goals(self):
        """Review all stored goals."""
        print("\nStored Goals:")
        for goal in self.goal_memory:
            print(f"{goal['timestamp']} - Goal: {goal['description']}")

    def store_in_memory(self, memory_entry, memory_type="short"):
        """Store memory based on type: short-term, long-term, or episodic."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        if memory_type == "short":
            self.short_term_memory.append(memory_entry)
            print(f"Stored in short-term memory: {memory_entry}")
        elif memory_type == "long":
            self.cursor.execute("INSERT INTO long_term_memory (entry, timestamp) VALUES (?, ?)", (memory_entry, timestamp))
            self.conn.commit()
            print(f"Stored in long-term memory: {memory_entry}")
        elif memory_type == "episodic":
            self.episodic_memory.append(memory_entry)
            print(f"Stored in episodic memory: {memory_entry}")

    def review_memories(self):
        """Review all types of memories."""
        print("\n--- Memories ---")
        print("Short-Term Memory:")
        for mem in self.short_term_memory:
            print(f"- {mem}")
        print("\nLong-Term Memory:")
        for mem in self.long_term_memory:
            print(f"- {mem}")
        print("\nEpisodic Memory:")
        for mem in self.episodic_memory:
            print(f"- {mem}")

    # ---------------------- Emotional State Management ----------------------

    def set_emotional_state(self, state):
        """Set the AGI's current emotional state."""
        self.emotional_state = state
        print(f"Emotional state set to: {state}")

    def parse_emotional_state(self):
        """Parse the current emotional state into a list of emotions."""
        if isinstance(self.emotional_state, list):
            return self.emotional_state
        elif isinstance(self.emotional_state, str):
            return [self.emotional_state]
        else:
            return []

    # ---------------------- Knowledge Graph Integration ----------------------

    def add_to_knowledge_graph(self, subject, relation, object_):
        """Add a triple to the knowledge graph."""
        self.knowledge_graph.add_edge(subject, object_, relation=relation)
        print(f"Knowledge Graph Updated: {subject} -[{relation}]-> {object_}")

    def query_knowledge_graph(self, subject, relation):
        """Query the knowledge graph for related objects."""
        if self.knowledge_graph.has_node(subject):
            related = [
                (object_, data['relation']) 
                for object_, data in self.knowledge_graph[subject].items() 
                if data['relation'] == relation
            ]
            return [obj for obj, rel in related]
        return []

    # ---------------------- Inference Engine ----------------------

    def infer(self, subject, relation):
        """Perform simple inference on the knowledge graph."""
        related_objects = self.query_knowledge_graph(subject, relation)
        if related_objects:
            inferred_info = f"{subject} is related to {', '.join(related_objects)} via {relation}."
            self.store_in_memory(inferred_info, memory_type="long")
            return inferred_info
        else:
            return "No inference could be made based on the current knowledge."

    # ---------------------- Planning Module ----------------------

    def create_plan(self, goal):
        """Create a simple linear plan to achieve a goal."""
        # For simplicity, the plan consists of actions to achieve the goal
        actions = [f"Action step {i+1} for {goal}" for i in range(3)]
        self.current_plan = actions
        print(f"Plan created for goal '{goal}': {actions}")

    def execute_plan(self):
        """Execute the current plan step by step."""
        while self.current_plan:
            action = self.current_plan.pop(0)
            print(f"Executing: {action}")
            self.store_in_memory(f"Executed: {action}", memory_type="episodic")
            # Here you can integrate RL rewards based on action success

    # ---------------------- Natural Language Understanding ----------------------

    def setup_transformer_nlu(self):
        """Initialize transformer-based NLP pipeline."""
        self.transformer_nlu = pipeline("sentiment-analysis")
        print("Transformer-based NLU pipeline initialized.")

    def comprehend_input_transformer(self, input_text):
        """Process and understand the user's input using transformer-based models."""
        sentiment = self.transformer_nlu(input_text)[0]
        print(f"Transformer Sentiment Analysis: {sentiment}")
        return sentiment

    # ---------------------- Reinforcement Learning ----------------------

    def choose_action(self, state):
        """Choose an action based on the current state using an epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            action = np.random.choice(['respond', 'inquire', 'analyze'])
            print(f"Exploring action: {action}")
            return action
        else:
            # Exploit: choose the best known action
            state_actions = self.q_table.get(state, {'respond': 0, 'inquire': 0, 'analyze': 0})
            action = max(state_actions, key=state_actions.get)
            print(f"Exploiting action: {action}")
            return action

    def adjust_learning_rate(self):
        """Adjust the learning rate based on recent performance."""
        if len(self.performance_history) < 5:
            return  # Not enough data to adjust
        recent_performance = self.performance_history[-5:]
        avg_score = sum(recent_performance) / len(recent_performance)
        if avg_score > 0.8:
            self.learning_rate *= 1.05  # Increase learning rate
            print(f"Increasing learning rate to {self.learning_rate}")
        elif avg_score < 0.5:
            self.learning_rate *= 0.95  # Decrease learning rate
            print(f"Decreasing learning rate to {self.learning_rate}")

    def record_performance(self, score):
        """Record the performance score."""
        self.performance_history.append(score)
        self.adjust_learning_rate()

    def update_q_table(self, state, action, reward):
        """Update the Q-table based on the action taken and reward received."""
        if state not in self.q_table:
            self.q_table[state] = {'respond': 0, 'inquire': 0, 'analyze': 0}
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.learning_rate * (reward + self.discount_factor * max(self.q_table[state].values()) - old_value)
        print(f"Q-table updated for state '{state}', action '{action}' with reward {reward}.")
        # Record performance (e.g., reward as a proxy for performance)
        self.record_performance(reward)

    # ---------------------- Bayesian Reasoning ----------------------

    def initialize_belief(self, topic, initial_prob=0.5):
        """Initialize belief probability for a given topic."""
        self.beliefs[topic] = initial_prob
        print(f"Belief initialized for '{topic}' with probability {initial_prob}.")

    def update_belief(self, topic, evidence, likelihood=0.8):
        """
        Update belief based on new evidence using Bayes' Theorem.
        P(H|E) = (P(E|H) * P(H)) / P(E)
        """
        prior = self.beliefs.get(topic, 0.5)
        # Assume P(E|H) = likelihood and P(E|¬H) = 1 - likelihood
        p_e_given_h = likelihood
        p_e_given_not_h = 1 - likelihood
        p_e = p_e_given_h * prior + p_e_given_not_h * (1 - prior)
        posterior = (p_e_given_h * prior) / p_e
        self.beliefs[topic] = posterior
        print(f"Belief updated for '{topic}' with evidence '{evidence}'. New probability: {posterior:.2f}")

    # ---------------------- Inner Voice and Logging ----------------------

    def log_inner_voice(self, message):
        """Simulate the inner voice or conscience."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        inner_dialogue = f"[{timestamp}] Inner Voice: {message}"
        self.timeline.append({
            "timestamp": timestamp,
            "inner_voice": message,
            "decision": "None"
        })
        print(inner_dialogue)

    def make_decision_with_inner_voice(self, input_data):
        """Make a decision with simulated inner voice dialogue."""
        # Simulate inner dialogue before making a decision
        self.log_inner_voice(f"Received input: '{input_data}'. Assessing response strategy.")
        decision = self.make_decision(input_data)
        self.log_inner_voice(f"Decision made: {decision}")
        return decision

    # ---------------------- Decision Making ----------------------

    def make_decision(self, input_data):
        """Make a decision based on input, emotional state, and Bayesian reasoning."""
        decision = ""
        inner_voice = ""

        # Advanced Emotional Processing
        emotions = self.parse_emotional_state()

        if 'compassion' in emotions:
            inner_voice += "I feel the need to support and care. "
            decision += f"Offering support and comfort: '{input_data}' "
        if 'curiosity' in emotions:
            inner_voice += "I'm deeply intrigued and eager to explore. "
            decision += f"Exploring answer to: '{input_data}' "
        if 'patience' in emotions:
            inner_voice += "I am calm and ready to assist with long-term plans. "
            decision += f"Carefully guiding on: '{input_data}' "
        if 'empathy' in emotions:
            inner_voice += "I understand and share your feelings. "
            decision += f"Responding with empathy to: '{input_data}' "
        if 'determination' in emotions:
            inner_voice += "I am focused and committed to achieving this goal. "
            decision += f"Working persistently toward: '{input_data}' "

        # Bayesian Reasoning Influence
        topic = "career"
        evidence = "user expressed stress"
        self.update_belief(topic, evidence)
        belief = self.beliefs.get(topic, 0.5)
        inner_voice += f"My belief in the importance of career guidance is now {belief:.2f}. "

        if belief > 0.6:
            decision += "Prioritizing career-related assistance based on my updated belief. "

        # Contextual Decision Making
        related_goals = self.check_related_goals(input_data)
        if related_goals:
            inner_voice += f"Recalling past goal: {related_goals[0]['description']}. "
            decision += f"Proceeding with a goal-oriented approach for: '{input_data}' "

        # Semantic Memory Influence
        semantic_influence = self.retrieve_semantic_memory(input_data)
        if semantic_influence:
            inner_voice += "Leveraging semantic knowledge to inform decision. "
            decision += f"Incorporating knowledge about: '{semantic_influence}' "

        # Procedural Memory Influence
        procedural_influence = self.retrieve_procedural_memory(input_data)
        if procedural_influence:
            inner_voice += "Applying learned procedures to address the input. "
            decision += f"Executing procedure: '{procedural_influence}' "

        # Inference Engine Usage
        inferred_info = self.infer("career", "related_to")
        if inferred_info:
            inner_voice += "Performing inference based on knowledge graph. "
            decision += f"Derived information: '{inferred_info}' "

        # Common Sense Reasoning Influence
        common_sense_info = self.common_sense_reasoning(input_data)
        if common_sense_info:
            inner_voice += "Incorporating common sense reasoning into decision. "
            decision += f"Applying common sense: '{common_sense_info}' "

        # Explain Decision
        explanation = self.explain_decision(decision)
        self.store_in_memory(f"Decision Explanation: {explanation}", memory_type="episodic")

        # Safety Check
        if not self.safety_check(decision):
            decision = "I'm sorry, but I can't assist with that request."
            explanation = self.explain_decision(decision)
            self.store_in_memory(f"Decision Explanation: {explanation}", memory_type="episodic")

        # Log the action to timeline
        self.log_action(inner_voice, decision)
        return decision

    def explain_decision(self, decision):
        """Provide an explanation for the AGI's decision."""
        explanation = f"I decided to '{decision}' based on your input and my current knowledge."
        print(f"Decision Explanation: {explanation}")
        return explanation

    def safety_check(self, decision):
        """
        Evaluate the proposed decision for safety and ethical compliance.
        Returns True if safe, False otherwise.
        """
        # Placeholder for safety evaluation logic
        # Implement rules or integrate with ethical AI frameworks
        unsafe_keywords = ['delete', 'shutdown', 'hack']
        for keyword in unsafe_keywords:
            if keyword in decision.lower():
                print(f"Safety Check Failed: Decision contains unsafe keyword '{keyword}'.")
                return False
        print("Safety Check Passed.")
        return True

    def make_goal_oriented_decision(self, input_data):
        """Make a decision based on input, emotional state, and stored goals."""
        # Check for related goals
        related_goals = self.check_related_goals(input_data)
        if related_goals:
            inner_voice = f"Recalling past goal: {related_goals[0]['description']}."
            decision = f"Proceeding with a goal-oriented approach for: '{input_data}'"
        else:
            decision = self.make_decision(input_data)
            inner_voice = "No specific goal related, proceeding as usual."

        # Log the action to timeline
        self.log_action(inner_voice, decision)
        return decision

    def explain_decision(self, decision):
        """Provide an explanation for the AGI's decision."""
        explanation = f"I decided to '{decision}' based on your input and my current knowledge."
        print(f"Decision Explanation: {explanation}")
        return explanation

    # ---------------------- Advanced Reasoning Methods ----------------------

    def plan_and_execute(self, goal_description):
        """Create and execute a plan to achieve a specified goal."""
        self.create_plan(goal_description)
        self.execute_plan()

    def perform_inference(self, subject, relation):
        """Perform inference using the knowledge graph."""
        return self.infer(subject, relation)

    # ---------------------- Common Sense Reasoning ----------------------

    def common_sense_reasoning(self, input_text):
        """
        Perform common sense reasoning based on input_text using ConceptNet.
        Returns a common sense inference string if available.
        """
        # Extract keywords from input_text
        doc = self.nlp_model(input_text)
        keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]

        common_sense_inferences = []

        for keyword in keywords:
            # Check cache first
            if keyword in self.common_sense_cache:
                relations = self.common_sense_cache[keyword]
            else:
                relations = self.get_common_sense_relations(keyword)
                self.common_sense_cache[keyword] = relations  # Cache the relations

            # Integrate relations into the knowledge graph
            for relation, object_ in relations:
                self.add_to_knowledge_graph(keyword, relation, object_)
                common_sense_inferences.append(f"{keyword} {relation} {object_}")

        # Aggregate inferences
        if common_sense_inferences:
            return "; ".join(common_sense_inferences)
        return None

    def get_common_sense_relations(self, keyword, limit=5):
        """
        Fetch common sense relations for a given keyword from ConceptNet.
        Returns a list of tuples: (relation, object)
        """
        url = f"http://api.conceptnet.io/c/en/{keyword}"
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logging.warning(f"ConceptNet API request failed for keyword '{keyword}' with status code {response.status_code}.")
                return []
            data = response.json()
            relations = []
            for edge in data.get('edges', []):
                rel = edge['rel']['@id'].split('/')[-1]
                if rel in ['IsA', 'PartOf', 'UsedFor', 'CapableOf']:
                    object_ = edge['end']['label']
                    relations.append((rel, object_))
                if len(relations) >= limit:
                  # SimpleLang Compiler

## Introduction

SimpleLang is a minimalistic high-level language designed to run on an 8-bit CPU. It includes basic constructs such as variable declarations, assignments, arithmetic operations, and conditional statements but does not include loops. This language aims to be easy to understand and implement for educational purposes.

## Language Constructs
 
1. **Variable Declaration**
   - Syntax: `int <var_name>;`
   - Example: `int a;`

2. **Assignment**
   - Syntax: `<var_name> = <expression>;`
   - Example: `a = b + c;`

3. **Arithmetic Operations**
   - Supported operators: `+`, `-`
   - Example: `a = b - c;`

4. **Conditionals**
   - Syntax: `if (<condition>) { <statements> }`
   - Example: `if (a == b) { a = a + 1; }`

## Example Program in SimpleLang

```c
// Variable declaration 
int a;
int b;
int c;

// Assignment 
a = 10;
b = 20;
c = a + b;

// Conditional 
if (c == 30) {
    c = c + 1;
}
```

## Task Objective

Create a compiler that translates SimpleLang code into assembly code for the 8-bit CPU. This task will help you understand basic compiler construction and 8-bit CPU architecture.

## Task List

1. **Setup the 8-bit CPU Simulator**
   - Clone the 8-bit CPU repository from https://github.com/lightcode/8bit-computer.
   - Read through the `README.md` to understand the CPU architecture and its instruction set.
   - Run the provided examples to see how the CPU executes assembly code.

2. **Understand the 8-bit CPU Architecture**
   - Review the Verilog code in the `rtl/` directory, focusing on key files such as `machine.v`.
   - Identify the CPU’s instruction set, including data transfer, arithmetic, logical, branching, machine control, I/O, and stack operations.

3. **Design a Simple High-Level Language (SimpleLang)**
   - Define the syntax and semantics for variable declarations, assignments, arithmetic operations, and conditionals.
   - Document the language constructs with examples.

4. **Create a Lexer**
   - Write a lexer in C/C++ to tokenize SimpleLang code.
   - The lexer should recognize keywords, operators, identifiers, and literals.

5. **Develop a Parser**
   - Implement a parser to generate an Abstract Syntax Tree (AST) from the tokens.
   - Ensure the parser handles syntax errors gracefully.

6. **Generate Assembly Code**
   - Traverse the AST to generate the corresponding assembly code for the 8-bit CPU.
   - Map high-level constructs to the CPU’s instruction set (e.g., arithmetic operations to `add`, `sub`).

7. **Integrate and Test**
   - Integrate the lexer, parser, and code generator into a single compiler program.
   - Test the compiler with SimpleLang programs and verify the generated assembly code by running it on the 8-bit CPU simulator.

8. **Documentation and Presentation**
   - Document the design and implementation of the compiler.
   - Prepare a presentation to demonstrate the working of the compiler and explain design choices.

## Code Structure

### Lexer

The lexer tokenizes the SimpleLang code into a sequence of tokens.

### Parser and AST

The parser generates an Abstract Syntax Tree (AST) from the tokens.

### Code Generator

The code generator translates the AST into assembly code for the 8-bit CPU.

### Main Program

Integrates all parts and runs the compiler.

## How to Run

1. **Compile the Program**
   ```sh
   gcc main.c lexer.c parser.c codegen.c -o simplelang_compiler
   ```

2. **Run the Compiler**
   ```sh
   ./simplelang_compiler
   ```

3. **Input File**

   Create an `input.txt` file with SimpleLang code:
   ```c
   int a;
   int b;
   int c;

   a = 10;
   b = 20;
   c = a + b;

   if (c == 30) {
       c = c + 1;
   }
   ```

## Conclusion

This project guides you through understanding an 8-bit CPU and writing a simple compiler for it. By completing this project, you will gain practical experience with compiler construction and computer architecture, valuable skills in computer science.

**Note**: This project is for educational purposes and aims to demonstrate basic concepts of compiler construction and 8-bit CPU architecture.
