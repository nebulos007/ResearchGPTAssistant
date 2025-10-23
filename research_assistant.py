"""
Main ResearchGPT Assistant Class

TODO: Implement the following functionality:
1. Integration with Mistral API
2. Advanced prompt engineering techniques
3. Research query processing
4. Answer generation and verification
"""

from mistralai import Mistral
import json
import time
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

class ResearchGPTAssistant:
    def __init__(self, config, document_processor):
        """
        Initialize ResearchGPT Assistant
        
        TODO:
        1. Store configuration and document processor
        2. Initialize Mistral client
        3. Load prompt templates
        4. Set up conversation history
        """
        self.config = config
        self.doc_processor = document_processor
        
        # Initialize Mistral client
        self.mistral_client = Mistral(api_key=self.config.MISTRAL_API_KEY)  # Initialize MistralClient here
        
        # Initialize conversation tracking
        self.conversation_history = []

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Track API usage
        self.api_call_count = 0
        self.total_tokens_used = 0

        # Load prompt templates
        self.prompts = self._load_prompt_templates()

        self.logger.info("ResearchGPT Assistant initialized.")
    
    def _load_prompt_templates(self):
        """
        Load comprehensive prompt templates for different research tasks
        
        Returns:
            dict: Dictionary of prompt templates
        """
        prompts = {
            'chain_of_thought': """
You are a research assistant helping to analyze academic papers. Think through this step by step.

Research Question: {query}

Context from Research Papers:
{context}

Let's approach this systematically:

1. First, let me understand what the question is asking:
   [Analyze the question and identify key components]

2. Next, let me examine the relevant information from the papers:
   [Review and summarize key points from the context]

3. Now, let me reason through the answer step by step:
   [Provide detailed reasoning for each aspect]

4. Finally, let me synthesize a comprehensive answer:
   [Combine insights into a coherent response]

Please provide a thorough, step-by-step analysis that clearly shows your reasoning process.
            """,
            
            'self_consistency': """
You are a research assistant analyzing academic papers. I want you to approach this question from a different angle than previous attempts.

Research Question: {query}

Context from Research Papers:
{context}

Consider this question carefully and provide your analysis. Focus on:
- Different aspects or perspectives not covered in other responses
- Alternative interpretations of the available evidence
- Unique insights from the research papers
- Any limitations or gaps in the current understanding

Provide a comprehensive answer that offers a fresh perspective while remaining grounded in the evidence.
            """,
            
            'react_research': """
You are conducting a research analysis using a structured approach. Follow this format:

Research Topic: {query}

Available Information: {context}

Use this structured thinking process:

Thought: [What do I need to understand or find out next?]
Action: [What should I do - analyze, search for specific info, compare, etc.]
Observation: [What did I learn from that action?]

Repeat this process until you have sufficient information to answer the question comprehensively.

End with a final summary of your findings.
            """,
            
            'document_summary': """
You are an expert research assistant. Please provide a comprehensive summary of this document.

Document Content: {content}

Your summary should include:

1. **Main Research Question/Objective**: What is the paper trying to achieve?

2. **Methodology**: How did the researchers approach the problem?

3. **Key Findings**: What are the most important results or discoveries?

4. **Conclusions**: What do the authors conclude from their work?

5. **Limitations**: What limitations does the study have?

6. **Significance**: Why is this research important?

Please structure your response clearly and focus on the most important aspects.
            """,
            
            'qa_with_context': """
You are a knowledgeable research assistant with access to academic papers. Answer the following question based on the provided context.

Question: {query}

Relevant Context from Research Papers:
{context}

Guidelines for your response:
- Base your answer primarily on the provided context
- If the context doesn't fully address the question, acknowledge this
- Cite specific findings or claims when possible
- Be precise and avoid speculation beyond what the papers support
- If there are conflicting viewpoints in the papers, mention them

Please provide a comprehensive, evidence-based answer.
            """,
            
            'verify_answer': """
You are a research quality assessor. Please evaluate the following answer for accuracy and completeness.

Original Question: {query}

Generated Answer: {answer}

Supporting Context: {context}

Please assess:

1. **Accuracy**: Is the answer factually correct based on the context?
2. **Completeness**: Does it address all aspects of the question?
3. **Evidence Support**: Is the answer well-supported by the provided context?
4. **Clarity**: Is the answer clear and well-structured?
5. **Limitations**: Are there any gaps or limitations in the answer?

Based on your assessment, provide:
- A verification score (1-10)
- Specific areas for improvement
- An improved version of the answer if needed

Focus on making the answer as accurate and helpful as possible.
            """,
            
            'basic_qa': """
You are a helpful research assistant. Please answer the following question based on the provided context from academic papers.

Question: {query}

Context: {context}

Please provide a clear, concise answer based on the information available.
            """,
            
            'workflow_conclusion': """
Based on the research workflow conducted, do we have sufficient information to provide a comprehensive answer?

Research Question: {query}
Information Gathered: {observation}

Consider:
- Is the core question addressed?
- Are there major gaps in understanding?
- Would additional research significantly improve the answer?

Respond with: YES (sufficient) or NO (need more research)
Provide a brief explanation of your decision.
            """
        }
        return prompts
    
    def _call_mistral(self, prompt, temperature=None):
        """
        Make API call to Mistral
        
        TODO: Implement Mistral API call:
        1. Use configured temperature or provided temperature
        2. Handle API errors gracefully
        3. Return response text
        4. Log API usage
        
        Args:
            prompt (str): Prompt to send to Mistral
            temperature (float): Temperature for generation
            
        Returns:
            str: Generated response
        """
        if temperature is None:
            temperature = self.config.TEMPERATURE
        max_tokens = self.config.MAX_TOKENS
            
        # Prepare messages for Mistral
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Make the API call using CORRECTED method
            start_time = time.time()

            # FIXED: Use the correct method to call Mistral
            chat_response = self.mistral_client.chat.complete(
                model=self.config.MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Track API usage
            self.api_call_count += 1
            api_time = time.time() - start_time

            # Extract response text
            response_text = chat_response.choices[0].message.content

            # Update token usage if available
            if hasattr(chat_response, 'usage') and chat_response.usage:
                if hasattr(chat_response.usage, 'total_tokens'):
                    self.total_tokens_used += chat_response.usage.total_tokens
                elif hasattr(chat_response.usage, 'prompt_tokens') and hasattr(chat_response.usage, 'completion_tokens'):
                    self.total_tokens_used += chat_response.usage.prompt_tokens + chat_response.usage.completion_tokens
            
            # Log API call details
            self.logger.info(f"Mistral API call successful (took {api_time:.2f}s)")
            self.logger.debug(f"Prompt length: {len(prompt)} characters")
            self.logger.debug(f"Response length: {len(response_text)} characters")

            return response_text
        
        except AttributeError as e:
            # Handle specific API structure issues
            self.logger.error(f"Mistral API structure error: {str(e)}")
            self.logger.error("Attempting alternative API call method.")
        
            try:
                # Alternative API call method
                chat_response = self.mistral_client.chat(
                    model=self.config.MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                response_text = chat_response.choices[0].message.content
                self.api_call_count += 1
                self.logger.info("Mistral API call successful using alternative method.")
                return response_text
        
            except Exception as e2:
                self.logger.error(f"Error calling Mistral API with alternative method: {str(e2)}")
                return f"Error calling Mistral API: {str(e2)}"
        
        except Exception as e:
            self.logger.error(f"Error calling Mistral API: {str(e)}")
            # Provide a more helpful error message
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                return "API Error: Authentication failed. Please check your Mistral API key in config.py."
            elif "model" in str(e).lower():
                return "API Error: Model not found. Please verify your model name"
            else:
                return f"API Error: {str(e)}. Please check your Mistral API configuration."

    def chain_of_thought_reasoning(self, query: str, context_chunks: List[Tuple]) -> str:
        """
        Use Chain-of-Thought prompting for complex reasoning
        
        TODO: Implement CoT reasoning:
        1. Build prompt with CoT template
        2. Include relevant context from documents
        3. Ask model to think step by step
        4. Return reasoned response
        
        Args:
            query (str): Research question
            context_chunks (list): Relevant document chunks as (text, score, doc_id tuples)
            
        Returns:
            str: Chain-of-thought response
        """
        # Build context string from chunks
        context = self._build_context_from_chunks(context_chunks)

        # Build CoT prompt
        cot_prompt = self.prompts['chain_of_thought'].format(query=query, context=context)
        
        # Call Mistral with CoT prompt
        response = self._call_mistral(cot_prompt, temperature=self.config.COT_TEMPERATURE)
        
        # Add to conversation history
        self.conversation_history.append({
            'type': 'chain_of_thought',
            'query': query,
            'response': response,
            'timestamp': time.time()
        })

        return response
    
    def self_consistency_generate(self, query, context_chunks, num_attempts=3):
        """
        Generate multiple responses and select most consistent
        
        TODO: Implement self-consistency:
        1. Generate multiple responses to same query
        2. Compare responses for consistency
        3. Select or combine best elements
        4. Return final consolidated answer
        
        Args:
            query (str): Research question
            context_chunks (list): Relevant document chunks  
            num_attempts (int): Number of responses to generate
            
        Returns:
            str: Most consistent response
        """
        self.logger.info(f"Generating {num_attempts} responses for self-consistency check.")

        responses = []
        context = self._build_context_from_chunks(context_chunks)

        # Generate multiple responses
        for i in range(num_attempts):
            # Generate response with slight temperature variation
            temp_variation = self.config.TEMPERATURE + (i * 0.1)
            temp_variation = min(temp_variation, 1.0)  # Cap temperature at 1.0

            prompt = self.prompts['self_consistency'].format(query=query, context=context)

            response = self._call_mistral(prompt, temperature=temp_variation)  # Your implementation here
            responses.append(response)

            self.logger.debug(f"Generated response {i+1}/{num_attempts}")
        
        # Implement consistency checking and selection
        best_response = self._select_most_consistent_response(responses, query, context)  # Implement actual selection logic
        
        # Add to conversation history
        self.conversation_history.append({
            'type': 'self_consistency',
            'query': query,
            'all_responses': responses,
            'selected_response': best_response,
            'timestamp': time.time()
        })

        return best_response
    
    def _select_most_consistent_response(self, responses: List[str], query: str, context: str) -> str:
        """
        Select the most consistent response from multiple attempts

        Args:
            responses (list): List of generated responses
            query (str): Research question
            context (str): Relevant context

        Returns:
            str: Most consistent response
        """
        if len(responses) == 1:
            return responses[0]
        
        # Filter out error responses
        valid_responses = [resp for resp in responses if not resp.startswith("API Error:")]

        if not valid_responses:
            return responses[0]  # Return first response if all are errors
        
        if len(valid_responses) == 1:
            return valid_responses[0]
        
        # Simple heuristic: select the response with median length and most context references
        scored_responses = []

        for resp in valid_responses:
            score = 0

            # Length score (prefer moderate length)
            length_score = 1.0 - abs(len(resp) - 1000) / 2000 # Optimal length around 1000 chars
            length_score = max(0, length_score)  # Ensure non-negative

            # Context reference score (count references to key terms)
            key_terms = query.lower().split()
            context_reference_score = sum(1 for term in key_terms if term in resp.lower())

            # Structure score (prefer well-structured responses)
            structure_score = 0
            if '1.' in resp or '2.' in resp:
                structure_score += 0.5
            if len(resp.split('\n')) > 3:  # multiple paragraphs
                structure_score += 0.3

            total_score = length_score + (context_reference_score * 0.3) + structure_score
            scored_responses.append((resp, total_score))

        # Select response with highest score
        best_response = max(scored_responses, key=lambda x: x[1])[0]

        self.logger.info("Selected most consistent response based on scoring.")
        return best_response
    
    def react_research_workflow(self, query):
        """
        Implement ReAct prompting for structured research workflow
        
        TODO: Implement ReAct workflow:
        1. Thought: Analyze what information is needed
        2. Action: Search documents for relevant information
        3. Observation: Review found information
        4. Repeat until sufficient information gathered
        5. Final reasoning and conclusion
        
        Args:
            query (str): Research question
            
        Returns:
            dict: Complete research workflow with steps and final answer
        """
        workflow_steps = []
        
        # TODO: Implement ReAct loop
        max_steps = 5

        # Initial context gathering
        initial_context = self.doc_processor.find_similar_chunks(query, top_k=8)
        context_text = self._build_context_from_chunks(initial_context)
        for step in range(max_steps):
            self.logger.info(f"ReAct workflow step {step + 1}/{max_steps}")
            # Generate thought about what to do next
            thought_prompt = f"""
            Research Workflow Step {step + 1}

            Research Question: {query}

            Previous Steps: {json.dumps(workflow_steps, indent=2)}

            Available Context: {context_text[:2000]}

            What should I think about or analyze next to answer this research question?
            Provide a clear thought about what aspect needs attention.
            """
            thought = self._call_mistral(thought_prompt, temperature=0.3)

            # Determine action based on thought
            action_prompt = f"""
            Action prompt:
            Based on this thought: {thought}

            What specific action should I take? Choose from:
              - SEARCH: Look for specific information in the documents
              - ANALYZE: Examine and interpret findings
              - COMPARE: Compare different approaches or findings
              - SYNTHESIZE: Combine information to form conclusions
              - CONCLUDE: Summarize and finalize the answer
              
              Respond with just the action and brief description.
            """
            
            action = self._call_mistral(action_prompt, temperature=0.2)

            # Execute the action
            observation = self._execute_react_action(action, query, context_text, workflow_steps)

            # Record workflow step
            workflow_step = {
                "step": step + 1,
                "thought": thought.strip(),
                "action": action.strip(),
                "observation": observation.strip()
            }
            workflow_steps.append(workflow_step)

            # Check if workflow should conclude
            if self._should_conclude_workflow(observation, query) or "CONCLUDE" in action.upper():
                self.logger.info(f"ReAct workflow concluded at step {step + 1}")
                break

        # Generate final conclusion
        workflow_result = {
            "query": query,
            "workflow": workflow_steps,
            "final_answer": self._generate_final_conclusion(query, workflow_steps)
        }
        # Add to conversation history
        self.conversation_history.append({
            "type": "react_workflow",
            "query": query,
            "workflow": workflow_result,
            "timestamp": time.time()
        })
        return workflow_result
    
    def _execute_react_action(self, action: str, query: str, context: str, previous_steps: List[Dict]) -> str:
        """
        Execute a specific action in the ReAct workflow

        Args:
            action (str): Action to execute
            query (str): Research question
            context (str): Available context
            previous_steps (list): Previous workflow steps

        Returns:
            str: Observation from executing the action
        """
        action_upper = action.upper()

        if "SEARCH" in action_upper:
            # Search for more specific information
            search_terms = self._extract_search_terms_from_action(action)
            if search_terms:
                new_chunks = self.doc_processor.find_similar_chunks(search_terms, top_k=3)
                observation = f"Found additional information: {self._build_context_from_chunks(new_chunks)}"
            else:
                observation = "Searched existing documents but no new relevant information found."
        
        elif "ANALYZE" in action_upper:
            # Analyze existing context
            analysis_prompt = f"""
            Analyze the following information in the context of the research question: {query}
            
            Information to analyze: {context[:1500]}
            Provide key insights and patterns you observe.
            """
            observation = self._call_mistral(analysis_prompt, temperature=0.4)

        elif "COMPARE" in action_upper:
            # Compare different findings
            compare_prompt = f"""
            Compare different approaches, methods, or findings related to: {query}

            Available Information: {context[:1500]}

            Highlight similarities, differences, and relative strengths or weaknesses.
            """
            observation = self._call_mistral(compare_prompt, temperature=0.4)

        elif "SYNTHESIZE" in action_upper:
            # Synthesize information
            synthesize_prompt = f"""
            Synthesize the gathered information to form coherent conclusions about: {query}

            Previous steps: {json.dumps(previous_steps, indent=2)}
            Context: {context[:1000]}

            Provide integrated insights that combine multiple pieces of information.
            """
            observation = self._call_mistral(synthesize_prompt, temperature=0.3)

        else:
            # Default observation
            observation = f"Executed action: {action}. Continuing analysis of available information."
        
        return observation
    
    def _extract_search_terms_from_action(self, action: str) -> str:
        """
        Extract search terms from the action description

        Args:
            action (str): Action description
        Returns:
            str: Extracted search terms
        """
        # Simple extraction - look for quoted terms or keywords after "Search for"
        if "search for" in action.lower():
            parts = action.lower().split("search for")
            if len(parts) > 1:
                return parts[1].strip().strip('"\'')
            
        return ""
    
    def _should_conclude_workflow(self, observation: str, query: str) -> bool:
        """
        Determine if the ReAct workflow has sufficient information to conclude

        Args:
            observation (str): Latest observation from workflow step
            query (str): Research question

        Returns:
            bool: Whether to conclude the workflow
        """
        # Use Mistral to assess if we have enough information
        conclusion_prompt = self.prompts['workflow_conclusion'].format(query=query, observation=observation)
        decision = self._call_mistral(conclusion_prompt, temperature=0.1)

        # Parse decision
        return "YES" in decision.upper() or "SUFFICIENT" in decision.upper()

    def _generate_react_conclusion(self, query: str, workflow_steps: List[Dict], context: str) -> str:
        """
        Generate final conclusion from ReAct workflow steps

        Args:
            query (str): Research question
            workflow_steps (list): Completed workflow steps
            context (str): Available context
        
        Returns:
            str: Final conclusion
        """

        conclusion_prompt = f"""
        Based on the structured research workflow conducted, please provide a comprehensive answer to the research question: {query}

        Research question: {query}

        Workflow Steps completed:
        {json.dumps(workflow_steps, indent=2)}

        Original Context:
        {context[:1000]}...

        Please synthesize all findings into a clear, comprehensive answer that addresses the research question directly.
        """
        final_answer = self._call_mistral(conclusion_prompt, temperature=0.3)
        return final_answer
    
    def verify_and_edit_answer(self, answer, original_query, context):
        """
        Verify answer quality and suggest improvements
        
        TODO: Implement verification process:
        1. Check answer relevance to query
        2. Verify claims against provided context
        3. Suggest improvements if needed
        4. Return verified/improved answer
        
        Args:
            answer (str): Generated answer to verify
            original_query (str): Original research question
            context (str): Document context used
            
        Returns:
            dict: Verification results and improved answer
        """
        self.logger.info("Starting answer verification process.")

        # Build verification prompt
        verification_prompt = self.prompts['verify_answer'].format(
            query=original_query,
            answer=answer,
            context=context[:2000]  # Limit context length
        )

        verification_result = self._call_mistral(verification_prompt, temperature=0.2)

        # Parse verification result and extract improved answer
        improved_answer = self._extract_improved_answer(verification_result, answer)
        confidence_score = self._extract_confidence_score(verification_result)

        verification_data = {
            'original_answer': answer,
            'verification_result': verification_result,
            'improved_answer': improved_answer,
            'confidence_score': confidence_score,
            'verification_timestamp': time.time()
        }

        self.logger.info("Answer verification process completed with confidence score: {confidence_score}")

        return verification_data
    
    def _extract_improved_answer(self, verification_result: str, original_answer: str) -> str:
        """
        Extract improved answer from verification result

        Args:
            verification_result (str): Full verification output
            original_answer (str): Original generated answer

        Returns:
            str: Improved answer if found, else original answer
        """
        # Look for improved version in the verification result
        lines = verification_result.split('\n')
        improved_section = False
        improved_lines = []

        for line in lines:
            line = line.strip()
            if 'improved' in line.lower() and ('version' in line.lower() or 'answer' in line.lower()):
                improved_section = True
                continue
            elif improved_section and line.strip():
                # Skip lines that look like section headers
                if not (line.strip().startswith('**') or line.strip().startswith('#')):
                    improved_lines.append(line)
                
        if improved_lines:
            improved_answer = '\n'.join(improved_lines).strip()
            # Only return if it's meaningfully different and longer than original
            if len(improved_answer) > len(original_answer) * 0.5:
                return improved_answer
            
        return original_answer

    def _extract_confidence_score(self, verification_result: str) -> Optional[float]:
        """
        Extract confidence score from verification result

        Args:
            verification_result (str): Full verification output
        
        Returns:
            float: Confidence score between 0 and 1
        """
        # Look for numerical scores in the verification result
        import re

        # Look for patterns like "score: 8/10" or "8 out of 10"
        score_patterns = [
            r'score[:\s]+(\d+)(?:/10|\s*out\s*of\s*10)',
            r'(\d+)(?:/10|\s*out\s*of\s*10)'
            r'(\d+)\.(\d+)(?:/10|\s*out\s*of\s*10)'
        ]

        for pattern in score_patterns:
            match = re.findall(pattern, verification_result.lower())
            if match:
                try:
                    if isinstance(match[0], tuple):
                        score = float(match[0][0] + '.' + match[0][1])
                    else:
                        score = float(match[0])
                    return min(score / 10.0, 1.0) # Normalize to 0-1
                except ValueError:
                    continue

        # Default confidence based on presence of positive keywords
        positive_indicators = ['accurate', 'comprehensive', 'well-supported', 'clear']
        negative_indicators = ['inaccurate', 'incomplete', 'unsupported', 'unclear']

        positive_count = sum(1 for word in positive_indicators if word in verification_result.lower())
        negative_count = sum(1 for word in negative_indicators if word in verification_result.lower())

        # Heuristic confidence score
        confidence = 0.7 + (0.1 * positive_count) - (0.15 * negative_count)
        return max(0.0, min(confidence, 1.0))
    
    def _build_context_from_chunks(self, context_chunks: List[Tuple]) -> str:
        """
        Build a single context string from document chunks
        
        Args:
            context_chunks (list): List of (text, score, doc_id) tuples

        Returns:
            str: Combined context string
        """
        if not context_chunks:
            return "No relevant context found in documents."
        
        context_parts = []
        for i, (chunk_text, score, doc_id) in enumerate(context_chunks):
            context_parts.append(f"Source {i+1} (from Document ID: {doc_id}):\n{chunk_text}\n")

        return "\n".join(context_parts)

    def answer_research_question(self, query: str, use_cot: bool = True, 
                                 use_verification: bool = True,
                                 strategy: str = "auto") -> Dict[str, Any]:
        """
        Main method to answer research questions with multiple strategies
        
        TODO: Implement complete research answering pipeline:
        1. Find relevant document chunks
        2. Apply selected prompting strategy
        3. Generate initial answer
        4. Verify and improve if requested
        5. Return comprehensive response
        
        Args:
            query (str): Research question
            use_cot (bool): Whether to use Chain-of-Thought
            use_verification (bool): Whether to verify answer
            strategy (str): Strategy to use ("cot", "self_consistency", "react", "auto", "basic")
            
        Returns:
            dict: Complete research response
        """
        self.logger.info(f"Answering research question: {query[:100]} using strategy: {strategy}...")

        start_time = time.time()

        # Find relevant documents
        relevant_chunks = self.doc_processor.find_similar_chunks(query, top_k=6)

        if not relevant_chunks:
            return {
                'query': query,
                'answer': "No relevant information found in the document database to answer the question.",
                'relevant_documents': 0,
                'sources_used': [],
                'strategy_used': 'none',
                'time_taken': time.time() - start_time
            }
        
        # Determine strategy
        if strategy == "auto":
            strategy = self._select_best_strategy(query, relevant_chunks)

        # Generate answer based on strategy
        if strategy == "react":
            workflow_result = self.react_research_workflow(query)
            answer = workflow_result['final_answer']
            strategy_data = workflow_result
        elif strategy == "self_consistency":
            answer = self.self_consistency_generate(query, relevant_chunks)
            strategy_data = {'type': 'self_consistency', 'num_attempts': 3}
        elif strategy == "cot" or use_cot:
            answer = self.chain_of_thought_reasoning(query, relevant_chunks)
            strategy_data = {'type': 'chain_of_thought'}
        else:
            # Basic QA without CoT
            context = self._build_context_from_chunks(relevant_chunks)
            basic_prompt = self.prompts['basic_qa'].format(query=query, context=context)
            answer = self._call_mistral(basic_prompt)
            strategy_data = {'type': 'basic_qa'}
            
        # Verify answer if requested
        verification_data = None
        final_answer = answer

        if use_verification and answer and not answer.startswith("API Error:"):
            context_str = self._build_context_from_chunks(relevant_chunks)
            verification_data = self.verify_and_edit_answer(answer, query, context_str)
            final_answer = verification_data['improved_answer']

        # Compile complete response
        response = {
            'query': query,
            'relevant_documents': len(relevant_chunks),
            'answer': final_answer,
            'verification': verification_data,
            'sources_used': strategy,
            'strategy_data': strategy_data,
            'time_taken': time.time() - start_time,
            'api_call_count': self.api_call_count,
            'total_tokens_used': self.total_tokens_used,
            'timestamp': time.time()    
        }

        self.logger.info(f"Completed answering question in {response['time_taken']:.2f} seconds.")

        return response
        
    def _select_best_strategy(self, query: str, context_chunks: List[Tuple]) -> str:
        """
        Select the best strategy based on query complexity and context
        
        Args:
            query (str): Research question
            context_chunks (list): Relevant document chunks
            
        Returns:
            str: Selected strategy ("cot", "self_consistency", "react", "basic")
        """
        # Analyze query complexity
        query_lower = query.lower()

        # Complex analytical questions -> ReAct
        if any(word in query_lower for word in ['analyze', 'compare', 'assess', 'evaluate', 'contrast']):
            return "react"
        
        # Questions requiring multiple perspectives -> Self-Consistency
        if any(word in query_lower for word in ['different, various, multiple, perspectives, viewpoints']):
            return "self_consistency"
        
        # Complex reasoning questions -> CoT
        if any(word in query_lower for word in ['explain', 'why', 'how', 'reasoning', 'because']):
            return "cot"
        
        # Simple factual questions -> Basic QA
        if any(word in query_lower for word in ['what is', 'define', 'who is', 'when', 'where']):
            return "basic"
        
        # Default to CoT for general cases
        return "cot"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history
        
        Returns:
            list: Conversation history entries
        """
        return self.conversation_history
    
    def get_api_usage_stats(self) -> Dict[str, Any]:
        """
        Retrieve API usage statistics
        
        Returns:
            dict: API usage stats
        """
        return {
            'api_call_count': self.api_call_count,
            'total_tokens_used': self.total_tokens_used,
            'average_tokens_per_call': (self.total_tokens_used / max(1, self.api_call_count))
        }
            
    def clear_conversation_history(self):
        """
        Clear the conversation history
        """
        self.conversation_history = []
        self.logger.info("Conversation history cleared.")

"""
# Example usage and testing
if __name__ == "__main__":
    # This section can be used for quick testing of the ResearchAssistant class
    from config import Config
    from document_processor import DocumentProcessor
    import os

    #Initialize components
    config = Config()
    doc_processor = DocumentProcessor(config)

    # Test if Mistral client can be initialized
    try:
        assistant = ResearchGPTAssistant(config, doc_processor)
        print("ResearchGPTAssistant initialized successfully.")
        print(f"API Usage Stats: {assistant.get_api_usage_stats()}")
    except Exception as e:
        print(f"Error initializing ResearchGPTAssistant: {str(e)}")
        print("Please ensure your Mistral API key is set correctly in config.py.")
        exit(1)

    # Check if sample documents are available and load them
    sample_dir = "data/sample_papers"
    if os.path.exists(sample_dir):
        pdf_files = [f for f in os.listdir(sample_dir) if f.endswith('.pdf')]

        if pdf_files:
            print(f"Found {len(pdf_files)} sample PDF files for testing. Processing...")

            # Process documents (this "loads" them into the document processor)
            for pdf_file in pdf_files[:3]: # Limit to first 3 for quick testing
                pdf_path = os.path.join(sample_dir, pdf_file)
                doc_id = doc_processor.process_document(pdf_path)
                print(f"Processed document {pdf_file} with ID: {doc_id}")

            # Build search index (required for similarity search)
            print("\nBuilding search index...")
            doc_processor.build_search_index()

            # Get document stats
            stats = doc_processor.get_document_stats()
            total_docs = stats.get('total_documents', 0)
            total_chunks = stats.get('total_chunks', 0)
            print(f"Documents loaded: {total_docs}, Total chunks: {total_chunks}")

            # Now documents are loaded, we can test the research assistant
            print("\n" + "-"*50)
            print("Testing ResearchGPTAssistant with a sample research question...")
            print("-"*50)

            #Test different strategies
            test_queries = [
                "What are the main contributions of this research?",
                "What methodology was used in the study?",
                "What are the limitations mentioned?",
                "How does this work compare to previous approachest?"
            ]

            for query in test_queries:
                print(f"\nQuery: {query}")
                print("-" * 40)

                try:
                    # Test basic answer
                    response = assistant.answer_research_question(query, use_cot=True, use_verification=False, strategy="cot")

                    print(f"Answer: {response['answer']}...")
                    print(f"Sources used: {response['sources_used']}")
                    print(f"Strategy: {response['strategy_data']}")
                    print(f"Time taken: {response['time_taken']:.2f} seconds")

                except Exception as e:
                    print(f"Error processing query '{query}': {str(e)}")

                # Test different strategies on one query
                print(f"\n" + "="*50)
                print(f"Testing different strategies for query: {query}")
                print("="*50)
"""