"""
Streamlit web interface for ExplainDoku
"""

import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional
import time

from ..core.grid import Grid
from ..core.solver import Solver
from ..explain.trace import TraceRecorder
from ..explain.verbalizer import Verbalizer


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="🧩 ExplainDoku",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧩 ExplainDoku - Sudoku Solver with Human-Style Explanations")
    st.markdown("Solve Sudoku puzzles with step-by-step explanations of the techniques used.")
    
    # Initialize session state
    if 'current_grid' not in st.session_state:
        st.session_state.current_grid = None
    if 'solver' not in st.session_state:
        st.session_state.solver = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'steps' not in st.session_state:
        st.session_state.steps = []
    if 'verbalizer' not in st.session_state:
        st.session_state.verbalizer = Verbalizer()
    
    # Sidebar
    with st.sidebar:
        st.header("📝 Input Puzzle")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "File Upload", "Example Puzzles"]
        )
        
        grid_str = None
        
        if input_method == "Text Input":
            grid_str = st.text_area(
                "Enter 81-character grid string:",
                placeholder="530070000600195000098000060800060003400803001700020006060000280000419005000080079",
                height=100
            )
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader("Upload puzzle file", type=['txt'])
            if uploaded_file is not None:
                grid_str = uploaded_file.read().decode('utf-8').strip()
        
        elif input_method == "Example Puzzles":
            example_choice = st.selectbox(
                "Choose an example puzzle:",
                [
                    "Easy Puzzle",
                    "Medium Puzzle", 
                    "Hard Puzzle",
                    "Very Hard Puzzle"
                ]
            )
            
            examples = {
                "Easy Puzzle": "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
                "Medium Puzzle": "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
                "Hard Puzzle": "000000000000003085001020000000507000004000100090000000500000073002010000000040009",
                "Very Hard Puzzle": "800000000003600000070090200050007000000045700000100030001000068008500010090000400"
            }
            
            grid_str = examples.get(example_choice, "")
        
        # Load puzzle button
        if st.button("Load Puzzle", type="primary"):
            if grid_str:
                try:
                    grid = Grid.from_string(grid_str)
                    st.session_state.current_grid = grid
                    st.session_state.solver = Solver(grid)
                    st.session_state.current_step = 0
                    st.session_state.steps = []
                    st.success("Puzzle loaded successfully!")
                except ValueError as e:
                    st.error(f"Invalid grid format: {e}")
            else:
                st.error("Please provide a grid string.")
    
    # Main content
    if st.session_state.current_grid is not None:
        display_puzzle_interface()
    else:
        display_welcome_screen()


def display_puzzle_interface():
    """Display the main puzzle solving interface"""
    grid = st.session_state.current_grid
    solver = st.session_state.solver
    verbalizer = st.session_state.verbalizer
    
    # Header with puzzle info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Current Puzzle State")
    
    with col2:
        if st.button("Solve All", type="primary"):
            solve_all_puzzle()
    
    with col3:
        if st.button("Reset"):
            st.session_state.current_grid = None
            st.session_state.solver = None
            st.session_state.current_step = 0
            st.session_state.steps = []
            st.rerun()
    
    # Display current grid
    display_grid(grid)
    
    # Step-by-step controls
    st.subheader("Step-by-Step Solving")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Next Step"):
            take_next_step()
    
    with col2:
        if st.button("Previous Step"):
            if st.session_state.current_step > 0:
                st.session_state.current_step -= 1
                st.rerun()
    
    with col3:
        if st.button("Auto Solve"):
            auto_solve_puzzle()
    
    with col4:
        if st.button("Clear Steps"):
            st.session_state.steps = []
            st.session_state.current_step = 0
            st.rerun()
    
    # Display steps
    if st.session_state.steps:
        st.subheader("Solving Steps")
        
        # Step navigation
        if len(st.session_state.steps) > 1:
            step_num = st.slider(
                "Go to step:",
                1, len(st.session_state.steps), 
                st.session_state.current_step + 1
            ) - 1
            
            if step_num != st.session_state.current_step:
                st.session_state.current_step = step_num
                st.rerun()
        
        # Display current step
        if 0 <= st.session_state.current_step < len(st.session_state.steps):
            step = st.session_state.steps[st.session_state.current_step]
            display_step(step, st.session_state.current_step + 1)
        
        # Display all steps
        with st.expander("View All Steps"):
            for i, step in enumerate(st.session_state.steps):
                display_step(step, i + 1, compact=True)
    
    # Sidebar with puzzle info
    with st.sidebar:
        st.header("📊 Puzzle Information")
        
        # Basic stats
        filled_cells = len(grid.get_filled_cells())
        empty_cells = len(grid.get_empty_cells())
        
        st.metric("Filled Cells", filled_cells)
        st.metric("Empty Cells", empty_cells)
        st.metric("Progress", f"{filled_cells/81*100:.1f}%")
        
        # Difficulty estimate
        if solver:
            difficulty = solver.get_difficulty_estimate()
            st.info(f"Estimated Difficulty: {difficulty}")
        
        # Technique help
        st.header("🔍 Technique Help")
        technique = st.selectbox(
            "Select technique:",
            verbalizer.templates.get_all_techniques()
        )
        
        if technique:
            help_info = verbalizer.get_technique_help(technique)
            st.write(f"**{help_info['name']}** ({help_info['difficulty']})")
            st.write(help_info['description'])


def display_grid(grid: Grid):
    """Display the Sudoku grid"""
    # Create a styled grid display
    grid_data = []
    for row in range(9):
        row_data = []
        for col in range(9):
            value = grid.get_value(row, col)
            row_data.append(value if value is not None else "")
        grid_data.append(row_data)
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(grid_data, index=range(1, 10), columns=range(1, 10))
    
    # Apply styling
    def highlight_box(val):
        return 'background-color: #f0f0f0'
    
    styled_df = df.style.applymap(
        lambda x: 'background-color: #e6f3ff' if x != "" else 'background-color: white',
        subset=pd.IndexSlice[1:9, 1:9]
    )
    
    # Add borders for 3x3 boxes
    for i in range(1, 10):
        for j in range(1, 10):
            if i % 3 == 1 and j % 3 == 1:
                styled_df = styled_df.set_properties(
                    subset=pd.IndexSlice[i:i+2, j:j+2],
                    **{'border': '2px solid black'}
                )
    
    st.dataframe(styled_df, use_container_width=True)


def display_step(step, step_number: int, compact: bool = False):
    """Display a solving step"""
    if compact:
        st.write(f"**Step {step_number}**: {step.explanation}")
    else:
        with st.container():
            st.markdown(f"### Step {step_number}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Technique:** {step.technique.replace('_', ' ').title()}")
                if step.cell_position and step.value:
                    row, col = step.cell_position
                    st.write(f"**Action:** Place {step.value} at R{row+1}C{col+1}")
            
            with col2:
                st.write(f"**Explanation:** {step.explanation}")
                
                if step.eliminations:
                    elims = []
                    for (row, col), digit in step.eliminations:
                        elims.append(f"{digit} from R{row+1}C{col+1}")
                    st.write(f"**Eliminations:** {', '.join(elims)}")


def take_next_step():
    """Take the next solving step"""
    solver = st.session_state.solver
    if not solver:
        return
    
    # Get next step
    next_step = solver._get_next_human_step()
    if next_step is None:
        next_step = solver._get_next_search_step()
        if next_step is None:
            st.warning("No more steps possible!")
            return
    
    # Apply the step
    if next_step.cell_position and next_step.value:
        row, col = next_step.cell_position
        st.session_state.current_grid.set_value(row, col, next_step.value)
    
    # Add to steps
    st.session_state.steps.append(next_step)
    st.session_state.current_step = len(st.session_state.steps) - 1
    
    st.success(f"Applied {next_step.technique.replace('_', ' ').title()}")
    st.rerun()


def solve_all_puzzle():
    """Solve the entire puzzle"""
    solver = st.session_state.solver
    if not solver:
        return
    
    with st.spinner("Solving puzzle..."):
        # Create a new solver from the current UI grid to preserve manual progress
        temp_solver = Solver(st.session_state.current_grid.copy())
        result = temp_solver.solve(use_search=True)
        
        if result.success:
            st.session_state.current_grid = result.final_grid
            st.session_state.steps = result.steps
            st.session_state.current_step = len(result.steps) - 1
            st.success("Puzzle solved successfully!")
        else:
            st.error("Could not solve puzzle completely.")
        
        st.rerun()


def auto_solve_puzzle():
    """Auto-solve step by step"""
    solver = st.session_state.solver
    if not solver:
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    step_count = 0
    max_steps = 50  # Prevent infinite loops
    
    while step_count < max_steps:
        if st.session_state.current_grid.is_solved():
            break
        
        next_step = solver._get_next_human_step()
        if next_step is None:
            next_step = solver._get_next_search_step()
            if next_step is None:
                break
        
        # Apply step
        if next_step.cell_position and next_step.value:
            row, col = next_step.cell_position
            st.session_state.current_grid.set_value(row, col, next_step.value)
        
        st.session_state.steps.append(next_step)
        step_count += 1
        
        # Update progress
        progress = min(step_count / max_steps, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Step {step_count}: {next_step.technique}")
        
        time.sleep(0.1)  # Small delay for visual effect
    
    progress_bar.empty()
    status_text.empty()
    
    if st.session_state.current_grid.is_solved():
        st.success(f"Auto-solved in {step_count} steps!")
    else:
        st.warning(f"Auto-solve stopped after {step_count} steps.")
    
    st.session_state.current_step = len(st.session_state.steps) - 1
    st.rerun()


def display_welcome_screen():
    """Display welcome screen when no puzzle is loaded"""
    st.markdown("""
    ## Welcome to ExplainDoku! 🧩
    
    ExplainDoku is a Sudoku solver that provides human-readable explanations for each step.
    
    ### Features:
    - **Human-style solving strategies**: Naked singles, hidden singles, locked candidates, pairs/triples, and more
    - **Step-by-step explanations**: Understand exactly why each move was made
    - **Interactive solving**: Watch the puzzle being solved step by step
    - **Multiple input methods**: Text input, file upload, or example puzzles
    
    ### How to get started:
    1. Use the sidebar to input a Sudoku puzzle
    2. Choose from text input, file upload, or example puzzles
    3. Click "Load Puzzle" to start solving
    4. Use "Next Step" to solve manually or "Solve All" for automatic solving
    
    ### Supported Techniques:
    - **Basic**: Naked singles, hidden singles
    - **Intermediate**: Locked candidates (pointing/claiming), naked/hidden pairs
    - **Advanced**: Naked/hidden triples, X-Wing, Swordfish, Jellyfish
    - **Search**: Backtracking with heuristics when human strategies fail
    """)
    
    # Show example puzzle
    st.subheader("Example Puzzle")
    example_grid = Grid.from_string("530070000600195000098000060800060003400803001700020006060000280000419005000080079")
    display_grid(example_grid)


if __name__ == "__main__":
    main()
