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

def solve_atomic_problem(client, model, problem, original_prompt):
    messages = [
        {"role": "user", "content": f"""This is a specific problem related to the original question: "{original_prompt}"
        
Solve this problem directly: {problem}

Be extremely concise and direct in your answer. Provide only the essential information needed to solve this specific problem. Keep your response brief and focused."""}
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

def should_decompose(client, model, problem, depth, max_depth):
    # Don't decompose if already at max depth
    if depth >= max_depth:
        return False
    
    messages = [
        {"role": "user", "content": f"""Consider this problem: {problem}

Is this problem best solved by breaking it down into smaller subproblems, or is it atomic enough to solve directly?
Use the following criteria:
- If the problem involves multiple distinct steps or components, it should be broken down
- If the problem is focused and specific enough to answer directly, it is atomic
- Consider the complexity of the problem

Answer with EXACTLY ONE WORD: either "DECOMPOSE" if it should be broken down or "ATOMIC" if it should be solved directly."""}
    ]
    
    response = client.chat.completions.create(model=model, messages=messages)
    decision = response.choices[0].message.content.strip().upper()
    
    return "DECOMPOSE" in decision

def break_down_problem(client, model, problem, original_prompt, depth=0):
    messages = [
        {"role": "user", "content": f"""I need to break down this problem into appropriate subproblems:

Problem: {problem}

Please identify 1-2 smaller subproblems that, when solved and combined, will answer the original problem. 
Choose the number of subproblems based on what makes the most sense for this specific problem.
Be very concise in how you formulate these subproblems.

Format your response EXACTLY as follows with ONLY these lines and nothing else:
Subproblem 1: [first subproblem - keep it brief and focused]
Subproblem 2: [second subproblem - only if needed]"""}
    ]
    
    breakdown = client.chat.completions.create(model=model, messages=messages)
    breakdown_text = breakdown.choices[0].message.content
    
    lines = [line.strip() for line in breakdown_text.split('\n') if line.strip()]
    
    subproblems = []
    for line in lines:
        if line.lower().startswith("subproblem"):
            # Extract the subproblem text after the "Subproblem N:" prefix
            colon_pos = line.find(":")
            if colon_pos != -1:
                subproblem = line[colon_pos + 1:].strip()
                subproblems.append(subproblem)
    
    if not subproblems:
        # Fallback: treat each line as a subproblem if parsing failed
        subproblems = lines[:2]  # Limit to 2 subproblems max
    
    print(f"Extracted {len(subproblems)} subproblems at depth: {depth}")
    return subproblems

def combine_solutions(client, model, original_problem, subproblems, sub_solutions, original_prompt):
    # Format all subproblems and their solutions for the combination prompt
    subproblem_solutions = ""
    for i, (subproblem, solution) in enumerate(zip(subproblems, sub_solutions)):
        subproblem_solutions += f"Subproblem {i+1}: \"{subproblem}\"\n"
        subproblem_solutions += f"Solution {i+1}: {solution}\n\n"
    
    messages = [
        {"role": "user", "content": f"""I've broken a complex problem into subproblems and solved each one.

Original question: {original_prompt}
Current problem to solve: {original_problem}

{subproblem_solutions}

Using ALL of these solutions, provide a complete but concise answer to the current problem. 
Be direct and include only what's necessary. Keep your final response brief and to the point."""}
    ]
    
    final_response = client.chat.completions.create(model=model, messages=messages)
    return final_response.choices[0].message.content

def solve_problem(client, model, problem, original_prompt, depth=0, max_depth=5):
    print(f"Processing problem at depth {depth}: {problem[:100]}..." if len(problem) > 100 else f"Processing problem at depth {depth}: {problem}")
    
    # If we've reached max depth or the problem is atomic, solve directly
    if depth >= max_depth or not should_decompose(client, model, problem, depth, max_depth):
        solution = solve_atomic_problem(client, model, problem, original_prompt)
        return {
            'solution': solution,
            'atomic': True,
            'depth': depth,
            'subproblems': [],
            'sub_solutions': []
        }
    
    # Otherwise, break down the problem
    subproblems = break_down_problem(client, model, problem, original_prompt, depth)
    
    # Process all subproblems in parallel
    sub_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all subproblems to be processed in parallel
        future_to_subproblem = {
            executor.submit(solve_problem, client, model, subprob, original_prompt, depth + 1, max_depth): subprob
            for subprob in subproblems
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_subproblem):
            sub_results.append(future.result())
    
    # Organize subproblem solutions in the original order
    ordered_sub_results = []
    for subprob in subproblems:
        for result in sub_results:
            if any(s == subprob for s in [result.get('original_problem', '')] + result.get('subproblems', [])):
                ordered_sub_results.append(result)
                break
    
    # If we couldn't match all results, use original order
    if len(ordered_sub_results) != len(subproblems):
        ordered_sub_results = sub_results
    
    # Extract solutions
    sub_solutions = [result['solution'] for result in ordered_sub_results]
    
    # Combine solutions
    combined_solution = combine_solutions(client, model, problem, subproblems, sub_solutions, original_prompt)
    
    return {
        'solution': combined_solution,
        'atomic': False,
        'depth': depth,
        'original_problem': problem,
        'subproblems': subproblems,
        'sub_solutions': sub_solutions,
        'sub_results': ordered_sub_results
    }

def process_prompt(client, model, prompt, max_depth=4):
    result = solve_problem(client, model, prompt, prompt, 0, max_depth)
    return result

def print_results(result, indent=0):
    """Recursively print results in a tree structure"""
    indent_str = "  " * indent
    
    if result.get('atomic', False):
        print(f"{indent_str}[ATOMIC] {result.get('original_problem', 'Problem')}:")
        print(f"{indent_str}Solution: {result['solution']}")
    else:
        print(f"{indent_str}Problem: {result.get('original_problem', 'Root problem')}")
        print(f"{indent_str}Final solution: {result['solution']}")
        
        if 'subproblems' in result and result['subproblems']:
            print(f"{indent_str}Subproblems:")
            for i, sub_result in enumerate(result.get('sub_results', [])):
                print(f"{indent_str}  #{i+1}:")
                print_results(sub_result, indent + 2)

while True:
    try:
        user_input = input("\nEnter your question (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        max_depth = 4  # Default max depth
        result = process_prompt(client, model, user_input, max_depth)
        
        print("\n=== RESULTS ===")
        print_results(result)
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")