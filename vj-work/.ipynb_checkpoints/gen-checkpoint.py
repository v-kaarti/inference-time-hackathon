# SPDX-License-Identifier: Apache-2.0

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

def solve_subproblem(client, model, problem):
    messages = [
        {"role": "user", "content": f"Solve this subproblem: {problem}"}
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

def process_prompt(client, model, prompt):
    decomposition_messages = [
        {"role": "user", "content": f"""Break this problem into exactly two distinct subproblems:
        Problem: {prompt}
        Format your response as:
        Subproblem 1: [first subproblem]
        Subproblem 2: [second subproblem]"""}
    ]
    
    breakdown = client.chat.completions.create(model=model, messages=decomposition_messages)
    breakdown_text = breakdown.choices[0].message.content

    print(breakdown_text)
    
    # Extract subproblems
    lines = breakdown_text.split('\n')
    subproblem1 = lines[0].replace('Subproblem 1:', '').strip()
    subproblem2 = lines[1].replace('Subproblem 2:', '').strip()
    
    # Solve each subproblem
    solution1 = solve_subproblem(client, model, subproblem1)
    solution2 = solve_subproblem(client, model, subproblem2)
    
    # Combine solutions
    combination_messages = [
        {"role": "user", "content": f"""Combine these two solutions into a final answer:
        Problem: {prompt}
        Solution to Subproblem 1: {solution1}
        Solution to Subproblem 2: {solution2}"""}
    ]
    
    final_response = client.chat.completions.create(model=model, messages=combination_messages)
    return {
        'subproblems': [subproblem1, subproblem2],
        'sub_solutions': [solution1, solution2],
        'final_solution': final_response.choices[0].message.content
    }

while True:
    try:
        user_input = input("\nEnter your question (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        result = process_prompt(client, model, user_input)
        
        print("\nSubproblem 1:", result['subproblems'][0])
        print("Solution 1:", result['sub_solutions'][0])
        print("\nSubproblem 2:", result['subproblems'][1])
        print("Solution 2:", result['sub_solutions'][1])
        print("\nFinal Solution:", result['final_solution'])
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")