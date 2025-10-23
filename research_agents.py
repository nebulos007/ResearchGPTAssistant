"""
AI Research Agents for Specialized Tasks - IMPROVED VERSION

Complete implementation with rate limiting, fixed performance tracking,
and enhanced complex workflow parsing.
"""

import json
import time
import logging
import random
from typing import List, Dict, Any, Optional

class BaseAgent:
    def __init__(self, research_assistant):
        """
        Base class for all research agents with improved performance tracking
        
        Args:
            research_assistant: ResearchGPTAssistant instance
        """
        self.assistant = research_assistant
        self.agent_name = "BaseAgent"
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.agent_name}")
        
        # FIXED: Improved performance tracking
        self.tasks_completed = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0
        self.last_task_time = 0
        
        # Rate limiting settings
        self.min_delay_between_calls = 1.0  # Minimum delay between API calls
        self.last_api_call_time = 0
    
    def _call_mistral_with_rate_limit(self, prompt: str, temperature: Optional[float] = None, 
                                    max_retries: int = 3) -> str:
        """
        IMPROVEMENT 1: Call Mistral with rate limiting and retry logic
        
        Args:
            prompt: Prompt to send to Mistral
            temperature: Generation temperature
            max_retries: Maximum number of retry attempts
            
        Returns:
            str: Generated response
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting: ensure minimum delay between calls
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call_time
                
                if time_since_last_call < self.min_delay_between_calls:
                    sleep_time = self.min_delay_between_calls - time_since_last_call
                    self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                
                # Make API call
                response = self.assistant._call_mistral(prompt, temperature)
                self.last_api_call_time = time.time()
                
                # Check if response indicates an error
                if response.startswith("API Error:"):
                    raise Exception(response)
                
                return response
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limiting specifically
                if "429" in error_str or "rate limit" in error_str.lower() or "capacity exceeded" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff for rate limiting
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error("Rate limit exceeded, max retries reached")
                        return "API Error: Rate limit exceeded. Please try again later."
                
                # Handle other API errors
                elif attempt < max_retries - 1:
                    delay = 1 + (attempt * 0.5)
                    self.logger.warning(f"API error, retrying in {delay:.2f}s: {error_str}")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"API call failed after {max_retries} attempts: {error_str}")
                    return f"API Error: {error_str}"
        
        return "API Error: Maximum retry attempts exceeded"
    
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Base method for executing agent tasks with improved tracking
        
        Args:
            task_input: Task parameters and input data
            
        Returns:
            dict: Task execution results
        """
        start_time = time.time()
        
        try:
            # This will be overridden by child classes
            result = self._execute_task_implementation(task_input)
            
            # FIXED: Update performance tracking
            execution_time = time.time() - start_time
            self.tasks_completed += 1
            self.total_execution_time += execution_time
            self.last_task_time = execution_time
            
            if result.get('success', False):
                self.successful_tasks += 1
            else:
                self.failed_tasks += 1
            
            return result
            
        except NotImplementedError:
            raise NotImplementedError("Each agent must implement _execute_task_implementation method")
        except Exception as e:
            execution_time = time.time() - start_time
            self.tasks_completed += 1
            self.failed_tasks += 1
            self.total_execution_time += execution_time
            self.last_task_time = execution_time
            
            return {
                'error': str(e),
                'success': False,
                'execution_time': execution_time
            }
    
    def _execute_task_implementation(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation method to be overridden by child classes"""
        raise NotImplementedError("Each agent must implement _execute_task_implementation method")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """FIXED: Get accurate agent performance statistics"""
        return {
            'agent_name': self.agent_name,
            'tasks_completed': self.tasks_completed,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.successful_tasks / max(1, self.tasks_completed),
            'total_execution_time': self.total_execution_time,
            'average_execution_time': self.total_execution_time / max(1, self.tasks_completed),
            'last_task_time': self.last_task_time
        }

class SummarizerAgent(BaseAgent):
    def __init__(self, research_assistant):
        """
        Agent specialized in document summarization with improved rate limiting
        """
        super().__init__(research_assistant)
        self.agent_name = "SummarizerAgent"
        
        # Summarization-specific settings
        self.max_summary_length = 1000
        self.include_methodology = True
        self.include_limitations = True
        
        self.logger.info("SummarizerAgent initialized")
    
    def _execute_task_implementation(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summarization task with proper performance tracking"""
        if 'doc_id' in task_input:
            return self.summarize_document(task_input['doc_id'])
        elif 'doc_ids' in task_input:
            return self.create_literature_overview(task_input['doc_ids'])
        else:
            return {
                "error": "Invalid task input for SummarizerAgent. Expected 'doc_id' or 'doc_ids'",
                "success": False
            }
    
    def summarize_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Summarize a specific document with rate limiting
        """
        try:
            # Get document data from document processor
            document_data = self.assistant.doc_processor.get_document_by_id(doc_id)
            
            if not document_data:
                return {
                    'error': f'Document {doc_id} not found',
                    'doc_id': doc_id,
                    'success': False
                }
            
            # Combine all chunks into full document text
            document_text = "\n\n".join(document_data['chunks'])
            
            # Limit text length for processing
            if len(document_text) > 8000:
                document_text = document_text[:8000] + "...[truncated]"
            
            # Create comprehensive summarization prompt
            summary_prompt = f"""
You are an expert research analyst. Please provide a comprehensive summary of this research document.

Document to analyze:
{document_text}

Please structure your summary as follows:

## 1. Research Objective
- What is the main research question or hypothesis?
- What problem is being addressed?

## 2. Methodology
- What research methods were used?
- What data or experiments were conducted?
- What are the key technical approaches?

## 3. Key Findings
- What are the most important results or discoveries?
- What evidence supports the main claims?
- What are the quantitative results if any?

## 4. Conclusions and Implications
- What do the authors conclude from their work?
- What are the broader implications for the field?
- How does this advance our understanding?

## 5. Limitations and Future Work
- What limitations does the study acknowledge?
- What areas need further research?
- What questions remain unanswered?

## 6. Significance
- Why is this research important?
- How does it relate to existing work in the field?
- What impact might it have?

Provide a clear, concise summary that captures the essence of the research while being accessible to someone familiar with the field.
            """
            
            # IMPROVEMENT 1: Use rate-limited API call
            summary = self._call_mistral_with_rate_limit(summary_prompt, temperature=0.3)
            
            # Extract key topics using simple keyword extraction
            key_topics = self._extract_key_topics(document_text, summary)
            
            # Structure summary output
            summary_data = {
                'doc_id': doc_id,
                'document_title': document_data.get('title', doc_id),
                'summary': summary,
                'word_count': len(summary.split()),
                'key_topics': key_topics,
                'original_length': len(document_text),
                'compression_ratio': len(summary) / len(document_text),
                'summary_timestamp': time.time(),
                'success': True
            }
            
            self.logger.info(f"Successfully summarized document {doc_id}")
            return summary_data
            
        except Exception as e:
            self.logger.error(f"Error summarizing document {doc_id}: {str(e)}")
            return {
                'error': str(e),
                'doc_id': doc_id,
                'success': False
            }
    
    def _extract_key_topics(self, document_text: str, summary: str) -> List[str]:
        """Extract key topics from document and summary"""
        import re
        from collections import Counter
        
        # Combine document and summary for topic extraction
        combined_text = (document_text + " " + summary).lower()
        
        # Remove common words and extract meaningful terms
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'a', 'an'}
        
        # Extract words that are likely to be important terms
        words = re.findall(r'\b[a-z]{3,}\b', combined_text)
        filtered_words = [w for w in words if w not in common_words and len(w) > 3]
        
        # Count frequency and return top terms
        word_counts = Counter(filtered_words)
        key_topics = [word for word, count in word_counts.most_common(10) if count > 2]
        
        return key_topics[:8]  # Return top 8 topics
    
    def create_literature_overview(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Create comprehensive overview of multiple documents with rate limiting"""
        try:
            if not doc_ids:
                return {'error': 'No document IDs provided', 'success': False}
            
            # Handle special case for 'all_available'
            if doc_ids == ['all_available']:
                doc_ids = list(self.assistant.doc_processor.documents.keys())
            
            # Summarize all documents first
            individual_summaries = []
            valid_summaries = []
            
            for doc_id in doc_ids:
                summary = self.summarize_document(doc_id)
                individual_summaries.append(summary)
                
                if summary.get('success', False):
                    valid_summaries.append(summary)
            
            if not valid_summaries:
                return {
                    'error': 'No valid summaries could be generated',
                    'individual_summaries': individual_summaries,
                    'success': False
                }
            
            # Combine summaries for overview analysis
            combined_summaries = "\n\n---\n\n".join([
                f"Document {s['doc_id']}:\n{s['summary']}" 
                for s in valid_summaries
            ])
            
            # Create comprehensive literature overview prompt
            overview_prompt = f"""
You are conducting a literature review analysis. Based on the summaries of multiple research papers below, provide a comprehensive overview.

Document Summaries:
{combined_summaries}

Please provide a literature overview with the following structure:

## 1. Common Research Themes
- What are the main research themes across these papers?
- What topics or problems are researchers focusing on?
- What are the recurring concepts or methodologies?

## 2. Methodological Approaches
- What different research methods are being used?
- Are there preferred approaches or techniques?
- How do the methodologies compare across studies?

## 3. Key Findings and Consensus
- What findings are consistent across multiple papers?
- Where do researchers agree on important points?
- What evidence is building toward scientific consensus?

## 4. Contradictions and Debates
- Where do the papers disagree or present conflicting results?
- What are the ongoing debates in the field?
- What controversies or unresolved questions exist?

## 5. Research Gaps and Opportunities
- What important questions remain unanswered?
- Where are the gaps in current knowledge?
- What areas need more research attention?

## 6. Future Research Directions
- What do the authors suggest for future work?
- What logical next steps emerge from this body of work?
- What new research questions are raised?

## 7. Field Evolution and Trends
- How is this research area evolving?
- What trends can be observed across the papers?
- What might be the future direction of the field?

Provide insights that synthesize across the papers rather than just summarizing each one individually.
            """
            
            # IMPROVEMENT 1: Use rate-limited API call
            overview = self._call_mistral_with_rate_limit(overview_prompt, temperature=0.4)
            
            # Analyze document topics across the collection
            all_topics = []
            for summary in valid_summaries:
                all_topics.extend(summary.get('key_topics', []))
            
            # Find most common topics across documents
            from collections import Counter
            topic_counts = Counter(all_topics)
            common_themes = [topic for topic, count in topic_counts.most_common(10) if count > 1]
            
            overview_data = {
                'overview': overview,
                'papers_analyzed': len(valid_summaries),
                'total_documents_attempted': len(doc_ids),
                'individual_summaries': individual_summaries,
                'common_themes': common_themes,
                'overview_timestamp': time.time(),
                'success': True
            }
            
            self.logger.info(f"Successfully created literature overview for {len(valid_summaries)} documents")
            return overview_data
            
        except Exception as e:
            self.logger.error(f"Error creating literature overview: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }

class QAAgent(BaseAgent):
    def __init__(self, research_assistant):
        """
        Agent specialized in question answering with improved performance tracking
        """
        super().__init__(research_assistant)
        self.agent_name = "QAAgent"
        
        # QA-specific settings
        self.default_chunk_count = 5
        self.confidence_threshold = 0.3
        
        self.logger.info("QAAgent initialized")
    
    def _execute_task_implementation(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: Execute QA task with proper performance tracking"""
        question = task_input.get('question', '')
        question_type = task_input.get('type', 'factual')
        
        if not question:
            return {
                'error': 'No question provided in task input',
                'success': False
            }
        
        if question_type == 'analytical':
            return self.answer_analytical_question(question)
        else:
            return self.answer_factual_question(question)
    
    def answer_factual_question(self, question: str) -> Dict[str, Any]:
        """Answer factual questions with rate limiting"""
        try:
            # Find relevant document chunks
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(
                question, top_k=self.default_chunk_count
            )
            
            if not relevant_chunks:
                return {
                    'question': question,
                    'answer': 'I could not find relevant information in the documents to answer this question.',
                    'sources': [],
                    'confidence': 'none',
                    'reasoning_type': 'factual',
                    'success': False
                }
            
            # Build context from relevant chunks
            context = self._build_qa_context(relevant_chunks)
            
            # Create factual QA prompt
            qa_prompt = f"""
You are a research assistant providing factual answers based on academic documents. Answer the question using only the information provided in the context.

Question: {question}

Context from Research Documents:
{context}

Instructions:
- Provide a clear, factual answer based solely on the information in the context
- If the context doesn't contain enough information, say so explicitly
- Cite specific sources when making claims
- Be precise and avoid speculation beyond what the documents support
- If there are multiple perspectives in the documents, mention them
- Rate your confidence level as: high, medium, low, or none

Please format your response as:

**Answer:** [Your factual answer]

**Confidence:** [high/medium/low/none]

**Sources Referenced:** [List the sources that support your answer]

**Limitations:** [Any limitations or gaps in the available information]
            """
            
            # IMPROVEMENT 1: Use rate-limited API call
            answer_response = self._call_mistral_with_rate_limit(qa_prompt, temperature=0.2)
            
            # Parse structured response
            parsed_response = self._parse_qa_response(answer_response)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(relevant_chunks, question, answer_response)
            
            result = {
                'question': question,
                'answer': parsed_response.get('answer', answer_response),
                'confidence': parsed_response.get('confidence', 'medium'),
                'confidence_score': confidence_score,
                'sources': [chunk[2] for chunk in relevant_chunks],
                'source_details': [{'doc_id': chunk[2], 'similarity': chunk[1]} for chunk in relevant_chunks],
                'limitations': parsed_response.get('limitations', ''),
                'reasoning_type': 'factual',
                'chunks_used': len(relevant_chunks),
                'success': True
            }
            
            self.logger.info(f"Successfully answered factual question: {question[:50]}...")
            return result
            
        except Exception as e:
            self.logger.error(f"Error answering factual question: {str(e)}")
            return {
                'question': question,
                'error': str(e),
                'success': False
            }
    
    def answer_analytical_question(self, question: str) -> Dict[str, Any]:
        """Answer analytical questions with rate limiting"""
        try:
            # Find relevant chunks for analytical reasoning
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(
                question, top_k=8  # More chunks for complex analysis
            )
            
            if not relevant_chunks:
                return {
                    'question': question,
                    'analysis': 'I could not find relevant information in the documents to provide analysis for this question.',
                    'reasoning_type': 'analytical',
                    'success': False
                }
            
            # Use Chain-of-Thought reasoning for complex analysis
            analysis_response = self.assistant.chain_of_thought_reasoning(question, relevant_chunks)
            
            # Enhance analysis with additional insights
            insights = self._generate_analytical_insights(question, relevant_chunks, analysis_response)
            
            result = {
                'question': question,
                'analysis': analysis_response,
                'reasoning_type': 'chain_of_thought',
                'analytical_insights': insights,
                'sources_analyzed': [chunk[2] for chunk in relevant_chunks],
                'depth_score': self._calculate_analysis_depth(analysis_response),
                'success': True
            }
            
            self.logger.info(f"Successfully provided analytical response for: {question[:50]}...")
            return result
            
        except Exception as e:
            self.logger.error(f"Error providing analytical answer: {str(e)}")
            return {
                'question': question,
                'error': str(e),
                'success': False
            }
    
    def _build_qa_context(self, chunks: List) -> str:
        """Build formatted context from document chunks for QA"""
        context_parts = []
        for i, (chunk_text, score, doc_id) in enumerate(chunks):
            context_parts.append(f"[Source {i+1} - {doc_id}]:\n{chunk_text}\n")
        return "\n".join(context_parts)
    
    def _parse_qa_response(self, response: str) -> Dict[str, str]:
        """Parse structured QA response into components"""
        import re
        
        parsed = {}
        
        # Extract answer
        answer_match = re.search(r'\*\*Answer:\*\*\s*(.*?)(?=\*\*|$)', response, re.DOTALL)
        if answer_match:
            parsed['answer'] = answer_match.group(1).strip()
        
        # Extract confidence
        conf_match = re.search(r'\*\*Confidence:\*\*\s*(high|medium|low|none)', response, re.IGNORECASE)
        if conf_match:
            parsed['confidence'] = conf_match.group(1).lower()
        
        # Extract limitations
        limit_match = re.search(r'\*\*Limitations:\*\*\s*(.*?)(?=\*\*|$)', response, re.DOTALL)
        if limit_match:
            parsed['limitations'] = limit_match.group(1).strip()
        
        return parsed
    
    def _calculate_confidence(self, chunks: List, question: str, answer: str) -> float:
        """Calculate confidence score based on source quality and relevance"""
        if not chunks:
            return 0.0
        
        # Base confidence on similarity scores
        avg_similarity = sum(chunk[1] for chunk in chunks) / len(chunks)
        
        # Adjust based on answer length and detail
        answer_quality = min(len(answer) / 500, 1.0)  # Normalize based on response length
        
        # Combine factors
        confidence = (avg_similarity * 0.7) + (answer_quality * 0.3)
        return min(confidence, 1.0)
    
    def _generate_analytical_insights(self, question: str, chunks: List, analysis: str) -> Dict[str, Any]:
        """Generate additional analytical insights"""
        return {
            'key_themes': self._extract_themes_from_analysis(analysis),
            'evidence_strength': 'strong' if len(chunks) >= 5 else 'moderate',
            'analysis_complexity': 'high' if len(analysis) > 1000 else 'medium',
            'sources_diversity': len(set(chunk[2] for chunk in chunks))
        }
    
    def _extract_themes_from_analysis(self, analysis: str) -> List[str]:
        """Extract key themes from analytical response"""
        themes = []
        theme_indicators = ['approach', 'method', 'finding', 'result', 'conclusion', 'limitation', 'implication']
        
        for indicator in theme_indicators:
            if indicator in analysis.lower():
                themes.append(indicator)
        
        return themes[:5]
    
    def _calculate_analysis_depth(self, analysis: str) -> str:
        """Calculate depth score for analytical response"""
        if len(analysis) > 2000:
            return 'comprehensive'
        elif len(analysis) > 1000:
            return 'detailed'
        elif len(analysis) > 500:
            return 'moderate'
        else:
            return 'basic'

class ResearchWorkflowAgent(BaseAgent):
    def __init__(self, research_assistant):
        """
        Agent for orchestrating complete research workflows with improved tracking
        """
        super().__init__(research_assistant)
        self.agent_name = "ResearchWorkflowAgent"
        
        # Initialize sub-agents for complex workflows
        self.summarizer = SummarizerAgent(research_assistant)
        self.qa_agent = QAAgent(research_assistant)
        
        # Workflow-specific settings
        self.max_questions = 5
        self.analysis_depth = 'comprehensive'
        
        self.logger.info("ResearchWorkflowAgent initialized with sub-agents")
    
    def _execute_task_implementation(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: Execute research workflow with proper performance tracking"""
        if 'research_topic' in task_input:
            return self.conduct_research_session(task_input['research_topic'])
        else:
            return {
                "error": "Invalid task input for ResearchWorkflowAgent. Expected 'research_topic'",
                "success": False
            }
    
    def conduct_research_session(self, research_topic: str) -> Dict[str, Any]:
        """Conduct comprehensive research session with rate limiting"""
        try:
            session_results = {
                'research_topic': research_topic,
                'generated_questions': [],
                'document_analysis': {},
                'question_answers': [],
                'research_gaps': '',
                'future_directions': '',
                'session_timestamp': time.time(),
                'success': True
            }
            
            self.logger.info(f"Starting research session on: {research_topic}")
            
            # Step 1: Generate relevant research questions
            questions_prompt = f"""
You are a research planning expert. Generate 3-5 specific, focused research questions about the following topic that can be answered using academic literature.

Research Topic: {research_topic}

Requirements for each question:
- Should be specific and answerable from research papers
- Should cover different aspects of the topic (what, how, why, implications, limitations)
- Should be suitable for academic analysis
- Should build upon each other logically

Please format as:
1. [Question about definition/nature]
2. [Question about methodology/approach]
3. [Question about findings/results]
4. [Question about implications/applications]
5. [Question about limitations/future work]

Provide only the numbered questions, one per line.
            """
            
            # IMPROVEMENT 1: Use rate-limited API call
            generated_questions_response = self._call_mistral_with_rate_limit(questions_prompt, temperature=0.4)
            
            # Parse generated questions
            questions = self._parse_research_questions(generated_questions_response)
            session_results['generated_questions'] = questions
            
            # Step 2: Find and analyze relevant documents
            relevant_docs = self.assistant.doc_processor.find_similar_chunks(research_topic, top_k=10)
            
            if relevant_docs:
                # Get unique document IDs
                doc_ids = list(set([doc[2] for doc in relevant_docs]))
                
                # Create literature overview using summarizer agent
                overview = self.summarizer.create_literature_overview(doc_ids)
                session_results['document_analysis'] = overview
                
                self.logger.info(f"Analyzed {len(doc_ids)} documents for research session")
            else:
                session_results['document_analysis'] = {
                    'overview': 'No relevant documents found for this research topic.',
                    'papers_analyzed': 0
                }
            
            # Step 3: Answer each generated question using QA agent
            question_answers = []
            for i, question in enumerate(questions[:self.max_questions]):
                self.logger.info(f"Answering research question {i+1}: {question[:50]}...")
                
                # Determine question type for appropriate answering strategy
                question_type = self._classify_question_type(question)
                
                # FIXED: Use the sub-agent's _execute_task_implementation method
                qa_result = self.qa_agent._execute_task_implementation({
                    'question': question,
                    'type': question_type
                })
                
                question_answers.append({
                    'question_number': i + 1,
                    'question': question,
                    'question_type': question_type,
                    'answer_data': qa_result
                })
            
            session_results['question_answers'] = question_answers
            
            # Step 4: Identify research gaps and future directions
            gaps_analysis = self._analyze_research_gaps(
                research_topic, 
                session_results['document_analysis'], 
                question_answers
            )
            
            session_results['research_gaps'] = gaps_analysis['gaps']
            session_results['future_directions'] = gaps_analysis['directions']
            
            # Step 5: Calculate session metrics
            session_results['session_metrics'] = self._calculate_session_metrics(session_results)
            
            self.logger.info(f"Completed research session on '{research_topic}'")
            
            return session_results
            
        except Exception as e:
            self.logger.error(f"Error conducting research session: {str(e)}")
            return {
                'research_topic': research_topic,
                'error': str(e),
                'success': False
            }
    
    def _parse_research_questions(self, questions_text: str) -> List[str]:
        """Parse numbered research questions from generated text"""
        import re
        
        questions = []
        lines = questions_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered questions
            match = re.match(r'^\d+\.\s*(.+)', line)
            if match:
                question = match.group(1).strip()
                if question and len(question) > 10:  # Filter out very short lines
                    questions.append(question)
        
        # If no numbered questions found, try to extract any question-like sentences
        if not questions:
            question_sentences = re.findall(r'[^.!?]*\?', questions_text)
            questions = [q.strip() for q in question_sentences if len(q.strip()) > 10]
        
        return questions[:5]  # Return at most 5 questions
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type for appropriate answering strategy"""
        question_lower = question.lower()
        
        # Analytical question indicators
        analytical_indicators = ['how', 'why', 'compare', 'analyze', 'evaluate', 'assess', 'examine', 'implications', 'relationship', 'impact']
        
        if any(indicator in question_lower for indicator in analytical_indicators):
            return 'analytical'
        else:
            return 'factual'
    
    def _analyze_research_gaps(self, topic: str, document_analysis: Dict, question_answers: List) -> Dict[str, str]:
        """Analyze research gaps and suggest future directions with rate limiting"""
        # Prepare context for gap analysis
        answered_questions = "\n".join([
            f"Q: {qa['question']}\nA: {qa['answer_data'].get('answer', 'No answer available')[:200]}..."
            for qa in question_answers
        ])
        
        gaps_prompt = f"""
Based on the research analysis conducted, identify research gaps and suggest future research directions.

Research Topic: {topic}

Document Analysis Summary:
{document_analysis.get('overview', 'No document analysis available')[:1000]}

Questions Explored:
{answered_questions}

Please provide:

## Research Gaps Identified:
- What important questions remain unanswered?
- Where are the limitations in current knowledge?
- What methodological gaps exist?
- What populations or contexts are understudied?

## Future Research Directions:
- What specific studies should be conducted next?
- What new methodologies could advance the field?
- What interdisciplinary approaches might be valuable?
- What practical applications need investigation?

Be specific and actionable in your recommendations.
        """
        
        # IMPROVEMENT 1: Use rate-limited API call
        gaps_response = self._call_mistral_with_rate_limit(gaps_prompt, temperature=0.4)
        
        # Parse gaps and directions (simplified)
        sections = gaps_response.split('## Future Research Directions:')
        gaps = sections[0].replace('## Research Gaps Identified:', '').strip()
        directions = sections[1].strip() if len(sections) > 1 else "Future research directions not clearly identified."
        
        return {
            'gaps': gaps,
            'directions': directions
        }
    
    def _calculate_session_metrics(self, session_results: Dict) -> Dict[str, Any]:
        """Calculate metrics for research session quality"""
        metrics = {}
        
        # Question coverage
        metrics['questions_generated'] = len(session_results.get('generated_questions', []))
        metrics['questions_answered'] = len(session_results.get('question_answers', []))
        
        # Document analysis
        doc_analysis = session_results.get('document_analysis', {})
        metrics['documents_analyzed'] = doc_analysis.get('papers_analyzed', 0)
        metrics['analysis_success'] = doc_analysis.get('success', False)
        
        # Answer quality
        successful_answers = sum(1 for qa in session_results.get('question_answers', []) 
                               if qa.get('answer_data', {}).get('success', False))
        metrics['answer_success_rate'] = successful_answers / max(1, metrics['questions_answered'])
        
        # Overall session quality
        if metrics['questions_answered'] >= 3 and metrics['documents_analyzed'] > 0:
            metrics['session_quality'] = 'comprehensive'
        elif metrics['questions_answered'] >= 2:
            metrics['session_quality'] = 'moderate'
        else:
            metrics['session_quality'] = 'basic'
        
        return metrics

class AgentOrchestrator:
    def __init__(self, research_assistant):
        """
        IMPROVEMENT 3: Enhanced orchestrator with improved workflow parsing
        """
        self.assistant = research_assistant
        
        # Initialize all specialized agents
        self.agents = {
            'summarizer': SummarizerAgent(research_assistant),
            'qa': QAAgent(research_assistant),
            'workflow': ResearchWorkflowAgent(research_assistant)
        }
        
        # Orchestrator settings
        self.task_history = []
        self.performance_stats = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.AgentOrchestrator")
        self.logger.info("AgentOrchestrator initialized with all specialized agents")
    
    def route_task(self, task_type: str, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Route tasks to appropriate agents with enhanced error handling"""
        start_time = time.time()
        
        try:
            # Validate task type
            if task_type not in self.agents:
                available_types = list(self.agents.keys())
                return {
                    "error": f"Unknown task type: {task_type}. Available types: {available_types}",
                    "success": False,
                    "available_agents": available_types
                }
            
            # Log task routing
            self.logger.info(f"Routing {task_type} task to {self.agents[task_type].agent_name}")
            
            # Execute task with appropriate agent
            agent = self.agents[task_type]
            result = agent.execute_task(task_input)
            
            # Add orchestrator metadata
            result['orchestrator_metadata'] = {
                'agent_used': agent.agent_name,
                'task_type': task_type,
                'routing_time': time.time() - start_time,
                'task_id': len(self.task_history) + 1
            }
            
            # Record task in history
            task_record = {
                'task_id': len(self.task_history) + 1,
                'task_type': task_type,
                'agent_used': agent.agent_name,
                'success': result.get('success', True),
                'execution_time': result.get('processing_time', time.time() - start_time),
                'timestamp': time.time()
            }
            self.task_history.append(task_record)
            
            # Update performance statistics
            self._update_performance_stats(task_type, task_record)
            
            self.logger.info(f"Successfully completed {task_type} task in {result['orchestrator_metadata']['routing_time']:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error routing {task_type} task: {str(e)}")
            return {
                "error": f"Orchestrator error for {task_type} task: {str(e)}",
                "success": False,
                "task_type": task_type,
                "routing_time": time.time() - start_time
            }
    
    def execute_complex_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """
        IMPROVEMENT 3: Enhanced complex workflow execution with better parsing
        """
        start_time = time.time()
        
        try:
            # Parse workflow requirements
            workflow_plan = self._parse_workflow_description(workflow_description)
            
            workflow_results = {
                'workflow_description': workflow_description,
                'workflow_plan': workflow_plan,
                'steps_executed': [],
                'final_result': {},
                'success': True
            }
            
            self.logger.info(f"Executing workflow with {len(workflow_plan['steps'])} steps")
            
            # Execute workflow steps
            for step_num, step in enumerate(workflow_plan['steps'], 1):
                self.logger.info(f"Executing workflow step {step_num}: {step['action']}")
                
                # Resolve dynamic parameters
                resolved_params = self._resolve_dynamic_parameters(step['parameters'])
                
                step_result = self.route_task(step['agent_type'], resolved_params)
                
                step_execution = {
                    'step_number': step_num,
                    'action': step['action'],
                    'agent_used': step['agent_type'],
                    'parameters': resolved_params,
                    'result': step_result,
                    'success': step_result.get('success', False)
                }
                
                workflow_results['steps_executed'].append(step_execution)
                
                # If a critical step fails, consider stopping the workflow
                if not step_result.get('success', False) and step.get('critical', False):
                    workflow_results['success'] = False
                    workflow_results['failure_reason'] = f"Critical step {step_num} failed: {step_result.get('error', 'Unknown error')}"
                    self.logger.error(f"Critical workflow step {step_num} failed")
                    break
            
            # Generate final synthesis if all steps completed successfully
            if workflow_results['success'] and workflow_results['steps_executed']:
                final_result = self._synthesize_workflow_results(workflow_results)
                workflow_results['final_result'] = final_result
            
            workflow_results['total_execution_time'] = time.time() - start_time
            
            self.logger.info(f"Completed complex workflow in {workflow_results['total_execution_time']:.2f}s")
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Error executing complex workflow: {str(e)}")
            return {
                'workflow_description': workflow_description,
                'error': str(e),
                'success': False,
                'execution_time': time.time() - start_time
            }
    
    def _parse_workflow_description(self, description: str) -> Dict[str, Any]:
        """
        IMPROVEMENT 3: Enhanced workflow parsing with better natural language understanding
        """
        description_lower = description.lower()
        steps = []
        
        # Enhanced keyword detection with more sophisticated patterns
        workflow_patterns = {
            'summarize': {
                'keywords': ['summarize', 'summary', 'overview', 'digest'],
                'variants': {
                    'multiple': ['multiple', 'all', 'literature', 'papers', 'documents'],
                    'single': ['document', 'paper', 'single', 'one']
                }
            },
            'question_answering': {
                'keywords': ['question', 'answer', 'what', 'how', 'why', 'explain'],
                'question_starters': ['what', 'how', 'why', 'when', 'where', 'who', 'which']
            },
            'research': {
                'keywords': ['research', 'analyze', 'investigate', 'study', 'examine', 'explore'],
                'depth_indicators': ['comprehensive', 'detailed', 'thorough', 'complete']
            }
        }
        
        # Step 1: Detect summarization requests
        if any(keyword in description_lower for keyword in workflow_patterns['summarize']['keywords']):
            if any(variant in description_lower for variant in workflow_patterns['summarize']['variants']['multiple']):
                steps.append({
                    'action': 'Create comprehensive literature overview',
                    'agent_type': 'summarizer',
                    'parameters': {'doc_ids': 'all_available'},
                    'critical': True,
                    'priority': 1
                })
            else:
                steps.append({
                    'action': 'Summarize primary document',
                    'agent_type': 'summarizer',
                    'parameters': {'doc_id': 'first_available'},
                    'critical': True,
                    'priority': 1
                })
        
        # Step 2: Detect question answering requests
        questions_found = []
        for starter in workflow_patterns['question_answering']['question_starters']:
            if starter in description_lower:
                # Extract potential question from description
                sentences = description.split('.')
                for sentence in sentences:
                    if starter in sentence.lower() and '?' in sentence:
                        questions_found.append(sentence.strip())
        
        # If explicit questions found, add QA steps
        for question in questions_found[:3]:  # Limit to 3 questions
            question_type = 'analytical' if any(word in question.lower() for word in ['how', 'why', 'analyze']) else 'factual'
            steps.append({
                'action': f'Answer question: {question[:50]}...',
                'agent_type': 'qa',
                'parameters': {
                    'question': question,
                    'type': question_type
                },
                'critical': False,
                'priority': 2
            })
        
        # If no explicit questions but QA keywords found, add general QA
        if not questions_found and any(keyword in description_lower for keyword in workflow_patterns['question_answering']['keywords']):
            # Generate questions based on the workflow description
            if 'methodology' in description_lower or 'method' in description_lower:
                steps.append({
                    'action': 'Answer question about methodology',
                    'agent_type': 'qa',
                    'parameters': {
                        'question': 'What methodology was used in the research?',
                        'type': 'analytical'
                    },
                    'critical': False,
                    'priority': 2
                })
            
            if 'contribution' in description_lower or 'finding' in description_lower:
                steps.append({
                    'action': 'Answer question about contributions',
                    'agent_type': 'qa',
                    'parameters': {
                        'question': 'What are the main contributions of this research?',
                        'type': 'analytical'
                    },
                    'critical': False,
                    'priority': 2
                })
        
        # Step 3: Detect research workflow requests
        if any(keyword in description_lower for keyword in workflow_patterns['research']['keywords']):
            # Extract research topic from description
            topic_keywords = []
            words = description.split()
            
            # Look for topic indicators
            topic_indicators = ['about', 'on', 'regarding', 'concerning', 'related to']
            for i, word in enumerate(words):
                if word.lower() in topic_indicators and i + 1 < len(words):
                    # Take next few words as topic
                    topic_keywords.extend(words[i+1:i+4])
                    break
            
            # If no explicit topic found, use key nouns from description
            if not topic_keywords:
                # Simple noun extraction (words that appear to be subjects)
                import re
                potential_topics = re.findall(r'\b[A-Za-z]{4,}\b', description)
                topic_keywords = potential_topics[:3]
            
            research_topic = ' '.join(topic_keywords) if topic_keywords else description[:50]
            
            # Determine if comprehensive research is needed
            is_comprehensive = any(indicator in description_lower for indicator in workflow_patterns['research']['depth_indicators'])
            
            if is_comprehensive:
                steps.append({
                    'action': 'Conduct comprehensive research session',
                    'agent_type': 'workflow',
                    'parameters': {'research_topic': research_topic},
                    'critical': True,
                    'priority': 3
                })
            else:
                steps.append({
                    'action': 'Conduct basic research analysis',
                    'agent_type': 'workflow',
                    'parameters': {'research_topic': research_topic},
                    'critical': True,
                    'priority': 2
                })
        
        # If no specific steps detected, create a default comprehensive workflow
        if not steps:
            steps.extend([
                {
                    'action': 'Create document overview',
                    'agent_type': 'summarizer',
                    'parameters': {'doc_ids': 'all_available'},
                    'critical': True,
                    'priority': 1
                },
                {
                    'action': 'Answer general research questions',
                    'agent_type': 'qa',
                    'parameters': {
                        'question': 'What are the main findings and contributions of this research?',
                        'type': 'analytical'
                    },
                    'critical': False,
                    'priority': 2
                }
            ])
        
        # Sort steps by priority
        steps.sort(key=lambda x: x.get('priority', 99))
        
        return {
            'description': description,
            'steps': steps,
            'estimated_time': len(steps) * 30,  # Rough estimate
            'complexity': self._assess_workflow_complexity(steps, description)
        }
    
    def _assess_workflow_complexity(self, steps: List[Dict], description: str) -> str:
        """Assess workflow complexity based on steps and description"""
        if len(steps) > 3:
            return 'high'
        elif len(steps) > 1 or any(step['agent_type'] == 'workflow' for step in steps):
            return 'medium'
        else:
            return 'low'
    
    def _resolve_dynamic_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        IMPROVEMENT 3: Resolve dynamic parameter values at runtime
        """
        resolved = parameters.copy()
        
        # Resolve 'all_available' document IDs
        if 'doc_ids' in resolved and resolved['doc_ids'] == 'all_available':
            available_docs = list(self.assistant.doc_processor.documents.keys())
            resolved['doc_ids'] = available_docs
        
        # Resolve 'first_available' document ID
        if 'doc_id' in resolved and resolved['doc_id'] == 'first_available':
            available_docs = list(self.assistant.doc_processor.documents.keys())
            resolved['doc_id'] = available_docs[0] if available_docs else None
        
        return resolved
    
    def _synthesize_workflow_results(self, workflow_results: Dict) -> Dict[str, Any]:
        """Enhanced workflow results synthesis"""
        # Collect all successful results
        successful_results = []
        for step in workflow_results['steps_executed']:
            if step['success']:
                successful_results.append(step['result'])
        
        # Create synthesis based on available results
        synthesis = {
            'key_insights': [],
            'combined_findings': '',
            'methodology_summary': '',
            'recommendations': '',
            'confidence_level': 'medium',
            'workflow_effectiveness': self._calculate_workflow_effectiveness(workflow_results)
        }
        
        # Extract insights from each successful step
        for result in successful_results:
            if 'summary' in result:
                synthesis['key_insights'].append(f"Summary insight: {result['summary'][:200]}...")
            elif 'answer' in result:
                synthesis['key_insights'].append(f"QA insight: {result['answer'][:200]}...")
            elif 'overview' in result:
                synthesis['key_insights'].append(f"Research insight: {result['overview'][:200]}...")
        
        # Generate combined findings summary
        if synthesis['key_insights']:
            insight_count = len(successful_results)
            synthesis['combined_findings'] = f"Workflow completed {insight_count} analysis steps successfully. Key findings include: " + "; ".join(synthesis['key_insights'][:3])
        
        return synthesis
    
    def _calculate_workflow_effectiveness(self, workflow_results: Dict) -> Dict[str, Any]:
        """Calculate effectiveness metrics for the workflow"""
        total_steps = len(workflow_results.get('steps_executed', []))
        successful_steps = sum(1 for step in workflow_results.get('steps_executed', []) if step.get('success', False))
        
        return {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'success_rate': successful_steps / max(1, total_steps),
            'overall_success': workflow_results.get('success', False)
        }
    
    def _update_performance_stats(self, task_type: str, task_record: Dict):
        """FIXED: Update performance statistics for agent types"""
        if task_type not in self.performance_stats:
            self.performance_stats[task_type] = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'total_time': 0,
                'average_time': 0,
                'success_rate': 0
            }
        
        stats = self.performance_stats[task_type]
        stats['total_tasks'] += 1
        
        if task_record['success']:
            stats['successful_tasks'] += 1
        
        stats['total_time'] += task_record['execution_time']
        stats['average_time'] = stats['total_time'] / stats['total_tasks']
        stats['success_rate'] = stats['successful_tasks'] / stats['total_tasks']
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator performance statistics with fixes"""
        # Collect individual agent stats
        agent_stats = {}
        for agent_type, agent in self.agents.items():
            agent_stats[agent_type] = agent.get_performance_stats()
        
        # Overall orchestrator stats
        total_tasks = len(self.task_history)
        successful_tasks = sum(1 for task in self.task_history if task['success'])
        
        overall_stats = {
            'total_tasks_orchestrated': total_tasks,
            'successful_tasks': successful_tasks,
            'overall_success_rate': successful_tasks / max(1, total_tasks),
            'average_task_time': sum(task['execution_time'] for task in self.task_history) / max(1, total_tasks),
            'task_distribution': self.performance_stats,
            'agent_performance': agent_stats
        }
        
        return overall_stats
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get complete task execution history"""
        return self.task_history
    
    def clear_history(self):
        """Clear task history and reset performance stats"""
        self.task_history = []
        self.performance_stats = {}
        
        # Also clear individual agent stats
        for agent in self.agents.values():
            agent.tasks_completed = 0
            agent.successful_tasks = 0
            agent.failed_tasks = 0
            agent.total_execution_time = 0
            agent.last_task_time = 0
        
        self.logger.info("Orchestrator history and all agent stats cleared")


""" # Example usage and testing with improvements
if __name__ == "__main__":
    from config import Config
    from document_processor import DocumentProcessor
    from research_assistant import ResearchGPTAssistant
    import os
    
    def main():
        #IMPROVED: Main function with better error handling and testing
        try:
            # Initialize the complete system
            config = Config()
            doc_processor = DocumentProcessor(config)
            research_assistant = ResearchGPTAssistant(config, doc_processor)
            
            # Initialize orchestrator with all agents
            orchestrator = AgentOrchestrator(research_assistant)
            
            print("AI Research Agents System (IMPROVED VERSION) initialized successfully!")
            print(f"Available agents: {list(orchestrator.agents.keys())}")
            print("Improvements: Rate limiting, Fixed performance tracking, Enhanced workflows")
            
            # Check if documents are available
            sample_dir = "data/sample_papers"
            if os.path.exists(sample_dir):
                pdf_files = [f for f in os.listdir(sample_dir) if f.endswith('.pdf')]
                
                if pdf_files:
                    print(f"\nProcessing {len(pdf_files)} documents...")
                    
                    # Process documents
                    for pdf_file in pdf_files[:2]:  # Process first 2 PDFs
                        pdf_path = os.path.join(sample_dir, pdf_file)
                        doc_processor.process_document(pdf_path)
                    
                    # Build search index
                    doc_processor.build_search_index()
                    stats = doc_processor.get_document_stats()
                    print(f"Documents loaded: {stats['num_documents']}")
                    
                    # Test improved system
                    print("\n" + "="*60)
                    print("TESTING IMPROVED AGENT SYSTEM")
                    print("="*60)
                    
                    # Test 1: Summarizer Agent with rate limiting
                    print("\n1. Testing Improved Summarizer Agent...")
                    doc_ids = list(doc_processor.documents.keys())
                    if doc_ids:
                        summary_result = orchestrator.route_task('summarizer', {
                            'doc_id': doc_ids[0]
                        })
                        print(f"   Summary success: {summary_result.get('success', False)}")
                        if summary_result.get('success'):
                            print(f"   Word count: {summary_result.get('word_count', 0)}")
                            print(f"   Key topics: {len(summary_result.get('key_topics', []))}")
                    
                    # Test 2: QA Agent with improved tracking
                    print("\n2. Testing Improved QA Agent...")
                    qa_result = orchestrator.route_task('qa', {
                        'question': 'What are the main methodological innovations?',
                        'type': 'analytical'
                    })
                    print(f"   QA success: {qa_result.get('success', False)}")
                    if qa_result.get('success'):
                        print(f"   Confidence: {qa_result.get('confidence', 'unknown')}")
                        print(f"   Sources used: {len(qa_result.get('sources', []))}")
                    
                    # Test 3: Enhanced Complex Workflow
                    print("\n3. Testing Enhanced Complex Workflow...")
                    enhanced_workflow_result = orchestrator.execute_complex_workflow(
                        "Provide a comprehensive analysis including document summaries and answer questions about the research methodology and key findings"
                    )
                    print(f"   Enhanced workflow success: {enhanced_workflow_result.get('success', False)}")
                    print(f"   Steps executed: {len(enhanced_workflow_result.get('steps_executed', []))}")
                    if enhanced_workflow_result.get('success'):
                        effectiveness = enhanced_workflow_result.get('final_result', {}).get('workflow_effectiveness', {})
                        print(f"   Workflow effectiveness: {effectiveness.get('success_rate', 0):.1%}")
                    
                    # Display improved performance statistics
                    print("\n" + "="*60)
                    print("IMPROVED PERFORMANCE STATISTICS")
                    print("="*60)
                    
                    stats = orchestrator.get_orchestrator_stats()
                    print(f"Total tasks: {stats['total_tasks_orchestrated']}")
                    print(f"Success rate: {stats['overall_success_rate']:.1%}")
                    print(f"Average time: {stats['average_task_time']:.2f}s")
                    
                    # Individual agent performance with fixes
                    for agent_type, perf in stats['agent_performance'].items():
                        if perf['tasks_completed'] > 0:
                            print(f"\n{agent_type.upper()} Agent (FIXED TRACKING):")
                            print(f"  Tasks completed: {perf['tasks_completed']}")
                            print(f"  Success rate: {perf['success_rate']:.1%}")
                            print(f"  Avg time: {perf['average_execution_time']:.2f}s")
                    
                    print(f"\n" + "="*60)
                    print("IMPROVEMENTS IMPLEMENTED:")
                    print("✓ Rate limiting with exponential backoff")
                    print("✓ Fixed performance tracking across all agents")
                    print("✓ Enhanced complex workflow parsing")
                    print("✓ Better error handling and recovery")
                    print("✓ Improved natural language understanding")
                    print("="*60)
                
                else:
                    print("No PDF files found. Please add research papers to test the improved agents.")
            
            else:
                print("Sample papers directory not found. Creating structure...")
                os.makedirs(sample_dir, exist_ok=True)
                print("Please add PDF files to data/sample_papers/ to test the improved system.")
        
        except Exception as e:
            print(f"Error in improved agent system testing: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run the main function
    main() """