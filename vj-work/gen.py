from openai import OpenAI
import concurrent.futures

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

def solve_subproblem(client, model, problem, original_prompt):
    messages = [
        {"role": "user", "content": f"""This is a subproblem of the original question: "{original_prompt}"
        
Solve this specific subproblem: {problem}

Be concise and direct in your answer. Provide only the necessary information to solve this specific subproblem."""}
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

def process_prompt(client, model, prompt):
    decomposition_messages = [
        {"role": "user", "content": f"""I need to break down this complex problem into exactly two distinct subproblems:

Problem: {prompt}

Please identify two smaller subproblems that, when solved and combined, will answer the original problem.

Format your response EXACTLY as follows with ONLY these two lines and nothing else:
Subproblem 1: [first subproblem]
Subproblem 2: [second subproblem]"""}
    ]
    
    breakdown = client.chat.completions.create(model=model, messages=decomposition_messages)
    breakdown_text = breakdown.choices[0].message.content
    
    lines = [line.strip() for line in breakdown_text.split('\n') if line.strip()]
    
    subproblem1 = None
    subproblem2 = None
    
    for line in lines:
        if line.lower().startswith("subproblem 1:"):
            subproblem1 = line[line.lower().find("subproblem 1:") + len("subproblem 1:"):].strip()
        elif line.lower().startswith("subproblem 2:"):
            subproblem2 = line[line.lower().find("subproblem 2:") + len("subproblem 2:"):].strip()
    
    if subproblem1 is None and subproblem2 is None and len(lines) >= 2:
        subproblem1 = lines[0]
        if "subproblem 1:" in subproblem1.lower():
            subproblem1 = subproblem1[subproblem1.lower().find("subproblem 1:") + len("subproblem 1:"):].strip()
        
        subproblem2 = lines[1]
        if "subproblem 2:" in subproblem2.lower():
            subproblem2 = subproblem2[subproblem2.lower().find("subproblem 2:") + len("subproblem 2:"):].strip()
    
    if not subproblem1 or not subproblem2:
        raise ValueError(f"Failed to extract subproblems from model response. Raw response: {breakdown_text}")
    
    print(f"Extracted subproblems:\n1: {subproblem1}\n2: {subproblem2}")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(solve_subproblem, client, model, subproblem1, prompt)
        future2 = executor.submit(solve_subproblem, client, model, subproblem2, prompt)
        
        solution1 = future1.result()
        solution2 = future2.result()
    
    combination_messages = [
        {"role": "user", "content": f"""I've broken a complex problem into subproblems and solved each one.

Original question: {prompt}

The first subproblem was: "{subproblem1}"
The solution to this first subproblem is:
{solution1}

The second subproblem was: "{subproblem2}"
The solution to this second subproblem is:
{solution2}

Using BOTH of these solutions, please provide a complete but concise answer to the original question. Be direct and include only what's necessary to address all aspects of the original question."""}
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