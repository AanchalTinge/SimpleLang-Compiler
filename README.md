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
