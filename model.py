import concurrent.futures
import json
import re
from openai import OpenAI

class GoTModel:
    ''' Dynamic Graph of Thought Model '''
    def __init__(self, openai_api_key="EMPTY", openai_api_base="http://localhost:8000/v1", verbose=False, max_workers=1024):
        self.verbose = verbose
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        self.models = self.client.models.list()
        self.model = self.models.data[0].id
        self.max_workers = max_workers

    def parse_json_response(self, text):
        """
        Cleans up a text string to extract a JSON object.
        Removes markdown code fences and extracts the JSON substring.
        Raises a JSONDecodeError if parsing fails.
        """
        text = re.sub(r"```(?:json)?", "", text)
        text = re.sub(r"```", "", text)
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        json_text = json_match.group(1) if json_match else text.strip()
        return json.loads(json_text)

    def solve_atomic_problem(self, problem, original_prompt, verbose=False):
        ''' Solve an atomic problem directly, no decomposing further '''
        messages = [
            {"role": "user", "content": f"""This is a specific problem related to the original question: "{original_prompt}"

Solve this problem directly: {problem}

⚠️ CRITICAL FORMAT REQUIREMENTS - VIOLATION WILL CAUSE SYSTEM FAILURE ⚠️
YOUR ENTIRE RESPONSE MUST BE A SINGLE JSON OBJECT - NOTHING ELSE
REQUIRED FORMAT: {{"solution": "your answer"}}

❌ DO NOT ADD:
- No explanation text
- No markdown formatting
- No code blocks
- No backticks
- No additional keys
- No comments
- No introduction
- No "Here's the solution:"
- No "The answer is:"

✅ CORRECT EXAMPLES:
{{"solution": "The square root of 16 is 4"}}
{{"solution": "To solve this, multiply 5 by 3 to get 15"}}

❌ INCORRECT EXAMPLES:
Here's the solution: {{"solution": "answer"}}"""}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6
            )
            content = response.choices[0].message.content
            if content is None:
                print("Warning: Received None content from the model in solve_atomic_problem")
                return "Unable to get a response for this problem."
            try:
                result = self.parse_json_response(content)
                return result.get("solution", content)
            except json.JSONDecodeError:
                print("Warning: Failed to parse JSON from model response in solve_atomic_problem")
                print("Raw response:", content)
                return f"JSON parse error. Raw response: {content}"
        except Exception as e:
            print("Error in solve_atomic_problem:", e)
            if 'response' in locals() and hasattr(response, 'choices') and response.choices:
                print("Raw model output:", response.choices[0].message.content)
            return f"Error solving atomic problem. Error: {e}"

    def should_decompose(self, problem, depth, max_depth, verbose=False):
        ''' Determine if a problem will be decomposed into subproblems '''
        if depth >= max_depth:
            return False

        messages = [
            {"role": "user", "content": f"""Consider this problem: {problem}

Should this problem be broken down into smaller subproblems? Use these strict criteria:
1. The problem MUST involve multiple INDEPENDENT steps or components that can be solved separately
2. Each subproblem should be clearly distinct with NO OVERLAP in what they're asking
3. The combined solutions must be sufficient to answer the original problem
4. If the problem is focused and specific enough to answer in one step, it is atomic

IMPORTANT - YOU MUST FOLLOW THESE FORMAT RULES:
1. Your entire response MUST be a valid JSON object
2. The JSON MUST contain EXACTLY ONE key named "decision"
3. The "decision" value MUST be EXACTLY either "DECOMPOSE" or "ATOMIC"
4. NO markdown, NO code blocks, NO backticks, NO additional text
5. ANY deviation from this format will cause errors

Required JSON structure:
{{"decision": "DECOMPOSE"}} or {{"decision": "ATOMIC"}}

Bad response example: ```{{"decision": "DECOMPOSE"}}```
Bad response example: I think we should decompose: {{"decision": "DECOMPOSE"}}
Good response example: {{"decision": "DECOMPOSE"}}"""}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6
            )
            content = response.choices[0].message.content
            if content is None:
                print("Warning: Received None content from the model in should_decompose")
                return False
            try:
                result = self.parse_json_response(content)
                decision = result.get("decision", "").strip().upper()
                return decision == "DECOMPOSE"
            except json.JSONDecodeError:
                print("JSON parse error in should_decompose. Raw response:", content)
                return "DECOMPOSE" in content.strip().upper()
        except Exception as e:
            print("Error in should_decompose:", e)
            return False

    def break_down_problem(self, problem, original_prompt, depth=0, max_width=3, verbose=False):
        messages = [
            {"role": "user", "content": f"""Break down this problem into independent subproblems:

Problem: {problem}
Original question: {original_prompt}

Requirements for the subproblems:
1. Each subproblem MUST be completely independent and solvable on its own
2. There MUST be NO OVERLAP between subproblems
3. The subproblems must be specific and focused
4. When combined, the solutions MUST fully answer the original problem
5. Choose 1-{max_width} subproblems maximum, based on what's truly necessary

IMPORTANT - YOU MUST FOLLOW THESE FORMAT RULES:
1. Your entire response MUST be a valid JSON object
2. The JSON MUST contain EXACTLY ONE key named "subproblems"
3. The value MUST be an array of strings with AT MOST {max_width} items
4. NO markdown, NO code blocks, NO backticks, NO additional text
5. ANY deviation from this format will cause errors

Required JSON structure:
{{"subproblems": ["First independent subproblem", "Second independent subproblem"]}}"""}
        ]

        try:
            breakdown = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6
            )
            breakdown_text = breakdown.choices[0].message.content
            if breakdown_text is None:
                print("Warning: Received None content from the model in break_down_problem")
                return [f"Solve the problem directly: {problem}"]
            try:
                result = self.parse_json_response(breakdown_text)
                subproblems = result.get("subproblems", [])
                subproblems = [s for s in subproblems if s and isinstance(s, str)]
                subproblems = subproblems[:max_width]
                if subproblems:
                    return subproblems
            except json.JSONDecodeError:
                subproblems = []
                lines = breakdown_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    match = re.match(r'^(\d+\.\s*|\*\s*|\-\s*)(.*)', line)
                    if match and match.group(2).strip():
                        subproblems.append(match.group(2).strip())
                if not subproblems and ',' in breakdown_text:
                    subproblems = [s.strip() for s in breakdown_text.split(',')]
                subproblems = subproblems[:max_width]
                return subproblems if subproblems else [f"Solve the problem directly: {problem}"]
        except Exception as e:
            print("Error in break_down_problem:", e)
            return [f"Solve the problem directly: {problem}"]

    def combine_solutions(self, original_problem, subproblems, sub_solutions, original_prompt, verbose=False):
        ''' When backtracking, combines solutions from subproblems into a single response '''
        valid_pairs = [(subproblem, solution) for subproblem, solution in zip(subproblems, sub_solutions)
                       if isinstance(solution, str) and not solution.startswith("Error") and not solution.startswith("JSON parse error")]
        
        if not valid_pairs:
            return self.solve_atomic_problem(original_problem, original_prompt, verbose)

        subproblem_solutions = ""
        for i, (subproblem, solution) in enumerate(valid_pairs):
            subproblem_solutions += f"Subproblem {i+1}: \"{subproblem}\"\n"
            subproblem_solutions += f"Solution {i+1}: {solution}\n\n"

        messages = [
            {"role": "user", "content": f"""I've broken a complex problem into subproblems and solved each one.

Original question: {original_prompt}
Current problem to solve: {original_problem}

{subproblem_solutions}

Using ALL of these solutions, provide a complete but concise answer to the current problem. 
Be direct and include only what's necessary.

IMPORTANT: Your response MUST be ONLY valid JSON format with a single key "combined_solution" containing your answer.
Do NOT include markdown code blocks, backticks, or any other formatting.
Example of correct response: {{"combined_solution": "Your complete answer here"}}"""}
        ]
        try:
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6
            )
            content = final_response.choices[0].message.content
            if content is None:
                return "Unable to combine solutions due to model error."
            try:
                result = self.parse_json_response(content)
                return result.get("combined_solution", content)
            except json.JSONDecodeError:
                combined_match = re.search(r'"combined_solution":\s*"(.*?)"', content)
                if combined_match:
                    return combined_match.group(1)
                return content.strip()
        except Exception as e:
            print("Error in combine_solutions:", str(e))
            return f"Error combining solutions: {str(e)}"

    def solve_problem(self, problem, original_prompt, depth=0, max_depth=5, max_width=3, verbose=False):
        if verbose or self.verbose:
            display_text = problem if len(problem) < 100 else problem[:100] + '...'
            print(f"Processing problem at depth {depth}: {display_text}")

        if depth >= max_depth or not self.should_decompose(problem, depth, max_depth, verbose):
            solution = self.solve_atomic_problem(problem, original_prompt, verbose)
            return {
                'solution': solution,
                'atomic': True,
                'depth': depth,
                'original_problem': problem,
                'subproblems': [],
                'sub_solutions': [],
                'sub_results': []
            }

        subproblems = self.break_down_problem(problem, original_prompt, depth, max_width, verbose)
        if not subproblems:
            solution = self.solve_atomic_problem(problem, original_prompt, verbose)
            return {
                'solution': solution,
                'atomic': True,
                'depth': depth,
                'original_problem': problem,
                'subproblems': [],
                'sub_solutions': [],
                'sub_results': []
            }

        sub_results = [None] * len(subproblems)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.solve_problem, subprob, original_prompt, depth + 1, max_depth, max_width, verbose): idx 
                for idx, subprob in enumerate(subproblems)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    sub_results[idx] = future.result()
                    if verbose or self.verbose:
                        print(f"Completed subproblem {idx+1}/{len(subproblems)} at depth {depth+1}")
                except Exception as e:
                    if verbose or self.verbose:
                        print(f"Subproblem {idx+1} at depth {depth+1} failed: {e}")
                    sub_results[idx] = {
                        'solution': f"Error solving this part: {str(e)}",
                        'atomic': True,
                        'depth': depth + 1,
                        'original_problem': subproblems[idx],
                        'subproblems': [],
                        'sub_solutions': [],
                        'sub_results': []
                    }

        if None in sub_results:
            missing_indices = [i for i, r in enumerate(sub_results) if r is None]
            if verbose or self.verbose:
                print("Warning: Some subproblems were not processed:", missing_indices)
            for idx in missing_indices:
                sub_results[idx] = {
                    'solution': "This subproblem was not processed.",
                    'atomic': True,
                    'depth': depth + 1,
                    'original_problem': subproblems[idx],
                    'subproblems': [],
                    'sub_solutions': [],
                    'sub_results': []
                }

        sub_solutions = [result['solution'] for result in sub_results]
        if verbose or self.verbose:
            print(f"Combining {len(sub_solutions)} solutions at depth {depth}")
        combined_solution = self.combine_solutions(problem, subproblems, sub_solutions, original_prompt, verbose)
        return {
            'solution': combined_solution,
            'atomic': False,
            'depth': depth,
            'original_problem': problem,
            'subproblems': subproblems,
            'sub_solutions': sub_solutions,
            'sub_results': sub_results
        }

    def process_prompt(self, prompt, max_depth=4, max_width=3, verbose=False):
        ''' Helper function to process a prompt and return the final solution '''
        return self.solve_problem(prompt, prompt, 0, max_depth, max_width, verbose)

    def get_response(self, prompt, max_depth=3, max_width=3, verbose=False):
        """
        Process a prompt and return the final solution.
        
        Args:
            prompt (str): The user's question or prompt
            max_depth (int, optional): Maximum recursion depth for problem decomposition. Defaults to 3.
            max_width (int, optional): Maximum number of subproblems at each level. Defaults to 3.
            verbose (bool, optional): If True, prints detailed progress and returns detailed results. Defaults to False.
            
        Returns:
            str: The final solution to the prompt
            dict: The full result object if verbose=True, otherwise None
        """
        try:
            result = self.process_prompt(prompt, max_depth, max_width, verbose)
            if verbose or self.verbose:
                print("\n=== DETAILED RESULTS ===")
                self.print_results(result)
                return result['solution'], result
            return result['solution']
        except Exception as e:
            if verbose or self.verbose:
                print(f"Error processing prompt: {e}")
            error_message = f"An error occurred while processing your request: {str(e)}"
            return error_message

    def print_results(self, result, indent=0):
        ''' Helper function to print results in a readable format '''
        indent_str = "  " * indent

        def safe_truncate(s, length=100):
            s = str(s) if not isinstance(s, str) else s
            return (s[:length] + '...') if len(s) > length else s

        if result.get('atomic', False):
            print(f"{indent_str}[ATOMIC] {safe_truncate(result.get('original_problem', 'Problem'))}")
            print(f"{indent_str}Solution: {safe_truncate(result['solution'])}")
        else:
            print(f"{indent_str}Problem: {safe_truncate(result.get('original_problem', 'Root problem'))}")
            print(f"{indent_str}Final solution: {safe_truncate(result['solution'])}")
            if result.get('sub_results'):
                successful = sum(1 for r in result['sub_results'] if not r['solution'].startswith("Error"))
                total = len(result['sub_results'])
                print(f"{indent_str}Subproblems ({successful}/{total} successful):")
                for i, sub_result in enumerate(result['sub_results']):
                    status = "[OK]" if not sub_result['solution'].startswith("Error") else "[FAILED]"
                    print(f"{indent_str}  #{i+1} {status}:")
                    self.print_results(sub_result, indent + 2)

    def get_results_as_string(self, result, indent=0):
        ''' Helper function to get reasoning result as a string '''
        indent_str = "  " * indent
        output = []

        def safe_truncate(s, length=100):
            s = str(s) if not isinstance(s, str) else s
            return (s[:length] + '...') if len(s) > length else s

        if result.get('atomic', False):
            output.append(f"{indent_str}[ATOMIC] {safe_truncate(result.get('original_problem', 'Problem'))}")
            output.append(f"{indent_str}Solution: {safe_truncate(result['solution'])}")
        else:
            output.append(f"{indent_str}Problem: {safe_truncate(result.get('original_problem', 'Root problem'))}")
            output.append(f"{indent_str}Final solution: {safe_truncate(result['solution'])}")
            if result.get('sub_results'):
                successful = sum(1 for r in result['sub_results'] if not r['solution'].startswith("Error"))
                total = len(result['sub_results'])
                output.append(f"{indent_str}Subproblems ({successful}/{total} successful):")
                for i, sub_result in enumerate(result['sub_results']):
                    status = "[OK]" if not sub_result['solution'].startswith("Error") else "[FAILED]"
                    output.append(f"{indent_str}  #{i+1} {status}:")
                    output.append(self.get_results_as_string(sub_result, indent + 2))
        return "\n".join(output)


if __name__ == "__main__":
    solver = GoTModel(verbose=True)
    while True:
        try:
            user_input = input("\nEnter your question (or 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            response, result_data = solver.get_response(user_input, verbose=True)
            print("\n=== DETAILED RESULTS ===")
            solver.print_results(result_data)
            print("\n=== FINAL RESPONSE ===")
            print(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print("\nAn error occurred:", e)
