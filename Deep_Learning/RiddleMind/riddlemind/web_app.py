"""
Streamlit Web Application for RiddleMind.

This module provides a beautiful web interface for the RiddleMind logic puzzle solver.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Optional

# Add the parent directory to the path to import riddlemind
sys.path.insert(0, str(Path(__file__).parent.parent))

from riddlemind.solver import RiddleMind, Solution


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="🧠 RiddleMind - Logic Puzzle Solver",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .puzzle-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .solution-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .constraint-item {
        background-color: #f8f9fa;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 3px solid #007bff;
    }
    .conclusion-item {
        background-color: #d4edda;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 3px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧠 RiddleMind</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Logic Puzzle Solver Bot</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Quick Examples")
        
        example_puzzles = {
            "Age Comparison": "Alice is older than Bob. Charlie is younger than Alice. Who is the oldest?",
            "Height Puzzle": "Alice is taller than Bob. Bob is taller than Charlie. Who is the shortest?",
            "Family Relations": "Alice is Bob's sister. Bob is Charlie's brother. What is Alice's relationship to Charlie?",
            "Spatial Arrangement": "Alice sits to the left of Bob. Bob sits to the left of Charlie. Who is in the middle?",
            "Complex Puzzle": "If Alice is older than Bob and Charlie is younger than Alice, then Bob is older than Charlie. Who is the youngest?"
        }
        
        selected_example = st.selectbox(
            "Choose an example:",
            list(example_puzzles.keys())
        )
        
        if st.button("Load Example"):
            st.session_state.puzzle_text = example_puzzles[selected_example]
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📝 Tips")
        st.markdown("""
        - Use proper names (capitalized)
        - Be specific about relationships
        - Include clear comparisons
        - Ask specific questions
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Logic Puzzle")
        
        # Text area for puzzle input
        puzzle_text = st.text_area(
            "Puzzle Text:",
            value=st.session_state.get("puzzle_text", ""),
            height=200,
            placeholder="Enter your logic puzzle here...\n\nExample:\nAlice is older than Bob.\nCharlie is younger than Alice.\nWho is the oldest?"
        )
        
        # Solve button
        col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
        
        with col1_2:
            solve_button = st.button("🧠 Solve Puzzle", type="primary", use_container_width=True)
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Or upload a text file:",
            type=['txt'],
            help="Upload a text file containing your logic puzzle"
        )
        
        if uploaded_file is not None:
            puzzle_text = uploaded_file.getvalue().decode("utf-8")
            st.session_state.puzzle_text = puzzle_text
            st.rerun()
    
    with col2:
        st.header("⚙️ Options")
        
        show_verbose = st.checkbox("Show detailed reasoning", value=False)
        show_prolog = st.checkbox("Show Prolog output", value=False)
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        if "last_solution" in st.session_state:
            solution = st.session_state.last_solution
            st.metric("Constraints", len(solution.parsed_constraints))
            st.metric("Entities", len(solution.parsed_constraints.entities))
            st.metric("Questions", len(solution.questions))
    
    # Solve the puzzle
    if solve_button and puzzle_text.strip():
        with st.spinner("🧠 RiddleMind is thinking..."):
            try:
                # Initialize solver
                solver = RiddleMind()
                
                # Solve the puzzle
                solution = solver.solve(puzzle_text.strip())
                
                # Store solution in session state
                st.session_state.last_solution = solution
                
                # Display results
                display_solution(solution, show_verbose, show_prolog)
                
            except Exception as e:
                st.error(f"❌ Error solving puzzle: {str(e)}")
                st.exception(e)
    
    # Display previous solution if available
    elif "last_solution" in st.session_state and st.session_state.get("puzzle_text"):
        st.markdown("---")
        st.header("📋 Previous Solution")
        display_solution(
            st.session_state.last_solution, 
            show_verbose, 
            show_prolog
        )


def display_solution(solution: Solution, show_verbose: bool, show_prolog: bool):
    """Display the solution in a formatted way."""
    
    # Original puzzle
    st.markdown('<div class="puzzle-box">', unsafe_allow_html=True)
    st.subheader("📝 Original Puzzle")
    st.write(solution.original_text)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Parsed constraints
    if solution.parsed_constraints:
        st.subheader("🔍 Parsed Constraints")
        
        for constraint in solution.parsed_constraints:
            st.markdown(f'<div class="constraint-item">• {constraint}</div>', unsafe_allow_html=True)
        
        if show_prolog:
            with st.expander("📋 Prolog Format"):
                st.code(solution.parsed_constraints.to_prolog(), language="prolog")
    else:
        st.warning("⚠️ No constraints were parsed from the input.")
    
    # Questions
    if solution.questions:
        st.subheader("❓ Questions")
        for question in solution.questions:
            st.write(f"• {question}")
    
    # Reasoning steps (if verbose)
    if show_verbose and solution.reasoning_steps:
        st.subheader("🧮 Reasoning Steps")
        for step in solution.reasoning_steps:
            with st.expander(f"Step {step.step_number}: {step.description}"):
                st.write(f"**Type:** {step.reasoning_type}")
                if step.derived_constraints:
                    st.write("**Derived Constraints:**")
                    for constraint in step.derived_constraints:
                        st.markdown(f'<div class="constraint-item">• {constraint}</div>', unsafe_allow_html=True)
    
    # Conclusions
    if solution.conclusions:
        st.markdown('<div class="solution-box">', unsafe_allow_html=True)
        st.subheader("✅ Conclusions")
        
        for conclusion in solution.conclusions:
            if not conclusion.startswith("Found") and not conclusion.startswith("Entities"):
                st.markdown(f'<div class="conclusion-item">• {conclusion}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Warnings
    if solution.warnings:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.subheader("⚠️ Warnings")
        for warning in solution.warnings:
            st.write(f"• {warning}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Errors
    if solution.errors:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.subheader("❌ Errors")
        for error in solution.errors:
            st.write(f"• {error}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary statistics
    if solution.parsed_constraints:
        st.subheader("📊 Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Constraints", len(solution.parsed_constraints))
        
        with col2:
            st.metric("Entities", len(solution.parsed_constraints.entities))
        
        with col3:
            st.metric("Questions", len(solution.questions))
        
        with col4:
            st.metric("Conclusions", len([c for c in solution.conclusions 
                                       if not c.startswith("Found") and not c.startswith("Entities")]))


if __name__ == "__main__":
    main()
