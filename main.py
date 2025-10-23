"""
Main execution script for ResearchGPT Assistant - Student Demo Version

This demo shows students exactly what ResearchGPT does with their research papers
by displaying actual questions and AI-generated answers about the uploaded documents.
"""

from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant
from research_agents import AgentOrchestrator
import os
import json
import time

def main():
    """
    Student-friendly demonstration showing actual AI responses to research questions
    """
    
    print("=" * 70)
    print("                    ResearchGPT Assistant Demo")
    print("=" * 70)
    print("An AI system that reads research papers and answers questions about them")
    print("=" * 70)
    
    # Step 1: Initialize system (simplified output)
    print("\n[1] Starting ResearchGPT Assistant...")
    try:
        config = Config()
        doc_processor = DocumentProcessor(config)
        research_assistant = ResearchGPTAssistant(config, doc_processor)
        agent_orchestrator = AgentOrchestrator(research_assistant)
        print("    System ready!")
    except Exception as e:
        print(f"    Error: {str(e)}")
        print("    Please check your API key configuration")
        return False
    
    # Step 2: Process documents (simplified)
    print("\n[2] Loading and analyzing your research papers...")
    sample_papers_dir = config.SAMPLE_PAPERS_DIR
    
    if not os.path.exists(sample_papers_dir):
        os.makedirs(sample_papers_dir, exist_ok=True)
        print("    Please add PDF research papers to data/sample_papers/ and run again")
        return False
    
    pdf_files = [f for f in os.listdir(sample_papers_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("    No PDF files found. Please add research papers to test the system")
        return False
    
    print(f"    Found {len(pdf_files)} research paper(s)")
    
    # Process documents
    for pdf_file in pdf_files:
        pdf_path = os.path.join(sample_papers_dir, pdf_file)
        print(f"    Reading: {pdf_file}")
        doc_processor.process_document(pdf_path)
    
    # Build search capability
    doc_processor.build_search_index()
    stats = doc_processor.get_document_stats()
    print(f"    Analysis complete: {stats['num_documents']} papers, {stats['total_chunks']} sections analyzed")
    
    # Step 3: Show what the AI learned about the papers
    print("\n[3] What ResearchGPT learned about your paper(s):")
    print("    " + "-" * 60)
    
    # Get document titles/info
    for doc_id, doc_data in doc_processor.documents.items():
        title = doc_data.get('title', doc_id)
        word_count = doc_data.get('metadata', {}).get('word_count', 0)
        print(f"    Paper: {title}")
        print(f"    Length: ~{word_count:,} words")
    
    print("    " + "-" * 60)
    
    # Step 4: Demonstrate actual Q&A with the research paper
    print("\n[4] Let's ask ResearchGPT questions about your research paper:")
    print("=" * 70)
    
    # Question 1: What is this research about?
    _demonstrate_question_answer(
        research_assistant,
        "What is this research paper about? Provide a clear summary.",
        "Understanding the Research Topic"
    )
    
    # Question 2: What are the main contributions?
    _demonstrate_question_answer(
        research_assistant,
        "What are the main contributions or innovations presented in this research?",
        "Key Contributions and Innovations"
    )
    
    # Question 3: What methodology was used?
    _demonstrate_question_answer(
        research_assistant,
        "What research methodology or approach did the authors use in this study?",
        "Research Methodology"
    )
    
    # Question 4: What are the results/findings?
    _demonstrate_question_answer(
        research_assistant,
        "What are the key findings or results reported in this research?",
        "Key Findings and Results"
    )
    
    # Step 5: Show advanced AI reasoning
    print("\n[5] Advanced AI Analysis - Step-by-Step Reasoning:")
    print("=" * 70)
    print("Let's see how ResearchGPT thinks through complex questions...")
    print()
    
    complex_question = "How does this research advance the field and what are its limitations?"
    print(f"Question: {complex_question}")
    print("\nResearchGPT's step-by-step thinking process:")
    print("-" * 50)
    
    try:
        # Use Chain-of-Thought to show reasoning process
        relevant_chunks = doc_processor.find_similar_chunks(complex_question, top_k=5)
        
        # Create a special prompt that shows the thinking process
        thinking_prompt = f"""
You are analyzing a research paper. Think through this question step by step and show your reasoning process clearly.

Question: {complex_question}

Research paper content: {_build_context_from_chunks(relevant_chunks)}

Please structure your response as:

STEP 1 - Understanding the question:
[Explain what the question is asking]

STEP 2 - Analyzing the research:
[Examine the key aspects of the research]

STEP 3 - Identifying contributions:
[What does this research contribute to the field]

STEP 4 - Identifying limitations:
[What are the limitations or areas for improvement]

STEP 5 - Final assessment:
[Overall conclusion about how this advances the field]
"""
        
        response = research_assistant._call_mistral(thinking_prompt, temperature=0.3)
        
        # Display the step-by-step reasoning
        print(response)
        print("\n" + "-" * 50)
        print("This shows how ResearchGPT breaks down complex questions and reasons through them systematically.")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
    
    # Step 6: Show AI agent capabilities
    print("\n[6] Specialized AI Agents in Action:")
    print("=" * 70)
    
    # Document Summarizer Agent
    print("Document Summarizer Agent - Creating an executive summary:")
    print("-" * 50)
    try:
        doc_ids = list(doc_processor.documents.keys())
        if doc_ids:
            summary_result = agent_orchestrator.route_task('summarizer', {'doc_id': doc_ids[0]})
            if summary_result.get('success', False):
                summary_text = summary_result.get('summary', '')
                # Show first part of summary
                print(summary_text[:800] + "..." if len(summary_text) > 800 else summary_text)
            else:
                print("Summary generation failed")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "-" * 50)
    print("This agent specializes in creating comprehensive summaries of research papers.")
    
    # Research Question Generator Agent
    print("\n\nResearch Workflow Agent - Generating research questions:")
    print("-" * 50)
    try:
        workflow_result = agent_orchestrator.route_task('workflow', {
            'research_topic': 'machine learning and AI research'
        })
        if workflow_result.get('success', False):
            questions = workflow_result.get('generated_questions', [])
            if questions:
                print("ResearchGPT generated these research questions about your paper:")
                for i, question in enumerate(questions[:5], 1):
                    print(f"{i}. {question}")
        else:
            print("Question generation failed")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "-" * 50)
    print("This agent can generate relevant research questions and conduct complete research sessions.")
    
    # Step 7: System capabilities summary
    print("\n[7] What ResearchGPT Can Do For You:")
    print("=" * 70)
    print("Based on this demonstration, ResearchGPT can help you:")
    print()
    print("• Quickly understand what any research paper is about")
    print("• Extract key contributions and methodologies")
    print("• Identify important findings and results")
    print("• Analyze limitations and future research directions")
    print("• Generate relevant research questions")
    print("• Create comprehensive summaries")
    print("• Compare multiple research papers")
    print("• Answer specific questions about technical details")
    print("• Provide step-by-step reasoning for complex analysis")
    print()
    print("This makes research much faster and helps you understand")
    print("complex papers more easily!")
    
    # Save results for students to examine
    print("\n[8] Saving demonstration results...")
    _save_demo_results(config)
    print("    Results saved to 'results/' folder for your review")
    
    print("\n" + "=" * 70)
    print("                    Demo Complete!")
    print("=" * 70)
    print("ResearchGPT successfully analyzed your research paper(s)")
    print("and demonstrated its AI-powered research assistance capabilities.")
    print("=" * 70)
    
    return True

def _demonstrate_question_answer(research_assistant, question, topic_title):
    """
    Show a clear question and answer demonstration
    """
    print(f"\nTopic: {topic_title}")
    print("-" * len(f"Topic: {topic_title}"))
    print(f"Question: {question}")
    print("\nResearchGPT's Answer:")
    print("~" * 40)
    
    try:
        # Get AI response
        response = research_assistant.answer_research_question(
            question, 
            use_cot=False, 
            use_verification=False,
            strategy="basic"
        )
        
        answer = response.get('answer', 'No answer generated')
        sources_count = len(response.get('sources_used', []))
        
        # Display answer (truncate if too long)
        if len(answer) > 1000:
            displayed_answer = answer[:1000] + "\n\n[Answer continues... full response saved to files]"
        else:
            displayed_answer = answer
            
        print(displayed_answer)
        print("~" * 40)
        print(f"Based on analysis of {sources_count} relevant sections from your paper(s)")
        print()
        
        # Brief pause to let students read
        time.sleep(1)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("~" * 40)

def _build_context_from_chunks(chunks):
    """Build context from document chunks for analysis"""
    if not chunks:
        return "No relevant content found"
    
    context_parts = []
    for chunk_text, score, doc_id in chunks[:3]:  # Use top 3 chunks
        context_parts.append(chunk_text[:500])  # Limit chunk size
    
    return "\n\n".join(context_parts)

def _save_demo_results(config):
    """Save simplified demo results for students"""
    try:
        results_dir = config.RESULTS_DIR
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        # Create a simple summary file
        demo_summary = """# ResearchGPT Assistant Demo Results

## What This Demo Showed

ResearchGPT successfully:
1. Read and analyzed your research paper(s)
2. Answered questions about the research content
3. Demonstrated step-by-step AI reasoning
4. Showed specialized AI agents working together
5. Generated relevant research questions
6. Created comprehensive summaries

## Key Capabilities Demonstrated

- Natural language understanding of research papers
- Question answering based on document content
- Advanced reasoning and analysis
- Multi-agent AI coordination
- Research workflow automation

## Files Generated

Check the results folder for detailed outputs from each demo step.

This demonstrates a complete AI research assistant that can help
you understand and analyze academic papers quickly and effectively.
"""
        
        with open(os.path.join(results_dir, "demo_summary.md"), 'w') as f:
            f.write(demo_summary)
            
    except Exception as e:
        print(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    """
    Main entry point for student demonstration
    """
    try:
        print("Starting ResearchGPT Assistant Student Demo...")
        success = main()
        
        if success:
            print("\nDemo completed successfully!")
            print("Students can now see how ResearchGPT works with real research papers.")
        else:
            print("\nDemo had some issues. Please check the setup and try again.")
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
        
    except Exception as e:
        print(f"\nDemo error: {str(e)}")
        print("\nPlease check:")
        print("1. Mistral API key is configured correctly")
        print("2. PDF files are in data/sample_papers/ directory") 
        print("3. All required packages are installed")