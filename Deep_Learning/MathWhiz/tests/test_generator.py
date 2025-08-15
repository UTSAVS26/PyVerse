import unittest
import sys
import os
import tempfile
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import MathExpressionGenerator
from model.tokenizer import MathTokenizer, create_tokenizer_from_data
from utils.math_solver import MathSolver

class TestMathExpressionGenerator(unittest.TestCase):
    """Test cases for MathExpressionGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = MathExpressionGenerator()
        self.solver = MathSolver()
    
    def test_generate_linear_equation(self):
        """Test linear equation generation."""
        expression, solution = self.generator.generate_linear_equation()
        
        # Check that expression contains 'x' and '='
        self.assertIn('x', expression)
        self.assertIn('=', expression)
        
        # Check that solution contains steps
        self.assertIn('Step', solution)
        
        # Basic verification - just check that it's a valid format
        self.assertIsInstance(expression, str)
        self.assertIsInstance(solution, str)
        self.assertGreater(len(expression), 0)
        self.assertGreater(len(solution), 0)
    
    def test_generate_polynomial_expression(self):
        """Test polynomial expression generation."""
        expression, solution = self.generator.generate_polynomial_expression()
        
        # Check that expression contains polynomial terms
        self.assertTrue('x' in expression or '(' in expression)
        
        # Check that solution contains steps
        self.assertIn('Step', solution)
        
        # Basic verification - just check that it's a valid format
        self.assertIsInstance(expression, str)
        self.assertIsInstance(solution, str)
        self.assertGreater(len(expression), 0)
        self.assertGreater(len(solution), 0)
    
    def test_generate_arithmetic_expression(self):
        """Test arithmetic expression generation."""
        expression, solution = self.generator.generate_arithmetic_expression()
        
        # Check that expression contains variables
        self.assertTrue('x' in expression or 'y' in expression)
        
        # Check that solution contains steps
        self.assertIn('Step', solution)
        
        # Basic verification - just check that it's a valid format
        self.assertIsInstance(expression, str)
        self.assertIsInstance(solution, str)
        self.assertGreater(len(expression), 0)
        self.assertGreater(len(solution), 0)
    
    def test_generate_dataset(self):
        """Test dataset generation."""
        num_samples = 100
        expressions, solutions = self.generator.generate_dataset(num_samples)
        
        # Check correct number of samples
        self.assertEqual(len(expressions), num_samples)
        self.assertEqual(len(solutions), num_samples)
        
        # Check that all expressions and solutions are strings
        for expr, sol in zip(expressions, solutions):
            self.assertIsInstance(expr, str)
            self.assertIsInstance(sol, str)
            self.assertGreater(len(expr), 0)
            self.assertGreater(len(sol), 0)
    
    def test_save_and_load_dataset(self):
        """Test dataset saving and loading."""
        # Generate small dataset
        expressions = ["2x + 3 = 9", "3x - 5 = 10", "x^2 + 5x + 6"]
        solutions = [
            "Step 1: Subtract 3 from both sides → 2x = 6 Step 2: Divide both sides by 2 → x = 3",
            "Step 1: Add 5 to both sides → 3x = 15 Step 2: Divide both sides by 3 → x = 5",
            "Step 1: Factor → (x + 2)(x + 3)"
        ]
        
        # Save dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data directory in temp dir
            data_dir = os.path.join(temp_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Temporarily change working directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                self.generator.save_dataset(expressions, solutions, "test_dataset.csv")
                
                # Check file exists
                self.assertTrue(os.path.exists("data/test_dataset.csv"))
                
                # Load dataset
                loaded_expressions, loaded_solutions = self.generator.load_dataset("test_dataset.csv")
                
                # Check loaded data matches original
                self.assertEqual(expressions, loaded_expressions)
                self.assertEqual(solutions, loaded_solutions)
            
            finally:
                os.chdir(original_cwd)
    
    def test_solve_linear_equation(self):
        """Test linear equation solving."""
        # Test case: 2x + 3 = 9
        solution = self.generator._solve_linear_equation(2, 3, 9)
        
        # Check solution contains expected steps
        self.assertIn("Subtract 3 from both sides", solution)
        self.assertIn("Divide both sides by 2", solution)
        self.assertIn("x = 3", solution)
    
    def test_expand_polynomial(self):
        """Test polynomial expansion."""
        # Test case: (2x + 1)(3x + 2)
        solution = self.generator._expand_polynomial("(2x + 1)(3x + 2)")
        
        # Check solution contains FOIL method
        self.assertIn("FOIL method", solution)
        self.assertIn("6x²", solution)  # 2*3 = 6
    
    def test_simplify_polynomial(self):
        """Test polynomial simplification."""
        # Test case: x^2 + 5x + 6
        solution = self.generator._simplify_polynomial("x^2 + 5x + 6")
        
        # Check solution contains factoring
        self.assertIn("Factor", solution)
    
    def test_simplify_arithmetic(self):
        """Test arithmetic simplification."""
        # Test case: 2x + 3y - 4x + 5y
        solution = self.generator._simplify_arithmetic("2x + 3y - 4x + 5y")
        
        # Check solution contains combining like terms
        self.assertIn("Combine like terms", solution)

class TestMathTokenizer(unittest.TestCase):
    """Test cases for MathTokenizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = MathTokenizer(vocab_size=100)
    
    def test_tokenize_math_expression(self):
        """Test mathematical expression tokenization."""
        expression = "2x + 3 = 9"
        tokens = self.tokenizer._tokenize_math_expression(expression)
        
        # Check that tokens are extracted correctly
        expected_tokens = ['2', 'x', '+', '3', '=', '9']
        self.assertEqual(tokens, expected_tokens)
    
    def test_tokenize_word(self):
        """Test word tokenization."""
        # Test step tokenization
        step_word = "Step 1:"
        tokens = self.tokenizer._tokenize_word(step_word)
        expected_tokens = ['Step', '1', ':']
        self.assertEqual(tokens, expected_tokens)
        
        # Test arrow tokenization
        arrow_word = "result → next"
        tokens = self.tokenizer._tokenize_word(arrow_word)
        self.assertIn('→', tokens)
    
    def test_fit_and_encode_decode(self):
        """Test tokenizer fitting, encoding, and decoding."""
        texts = [
            "2x + 3 = 9",
            "Step 1: Subtract 3 from both sides → 2x = 6",
            "Step 2: Divide both sides by 2 → x = 3"
        ]
        
        # Fit tokenizer
        self.tokenizer.fit(texts)
        
        # Test encoding
        encoded = self.tokenizer.encode("2x + 3 = 9")
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(token, int) for token in encoded))
        
        # Test decoding
        decoded = self.tokenizer.decode(encoded)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded), 0)
    
    def test_encode_batch(self):
        """Test batch encoding."""
        texts = ["2x + 3 = 9", "3x - 5 = 10"]
        self.tokenizer.fit(texts)
        
        batch_encoded = self.tokenizer.encode_batch(texts)
        self.assertEqual(len(batch_encoded), len(texts))
        self.assertTrue(all(isinstance(seq, list) for seq in batch_encoded))
    
    def test_save_and_load(self):
        """Test tokenizer saving and loading."""
        texts = ["2x + 3 = 9", "Step 1: Subtract 3 → 2x = 6"]
        self.tokenizer.fit(texts)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            # Save tokenizer
            self.tokenizer.save(tmp_file.name)
            
            # Create new tokenizer and load
            new_tokenizer = MathTokenizer()
            new_tokenizer.load(tmp_file.name)
            
            # Test that loaded tokenizer works
            test_text = "2x + 3 = 9"
            original_encoded = self.tokenizer.encode(test_text)
            loaded_encoded = new_tokenizer.encode(test_text)
            
            self.assertEqual(original_encoded, loaded_encoded)
            
            # Clean up - try to remove file, ignore if it fails
            try:
                os.unlink(tmp_file.name)
            except:
                pass

class TestMathSolver(unittest.TestCase):
    """Test cases for MathSolver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solver = MathSolver()
    
    def test_solve_linear_equation(self):
        """Test linear equation solving."""
        expression = "2x + 3 = 9"
        solution = self.solver.solve_step_by_step(expression)
        
        # Check solution contains steps
        self.assertIn("Step", solution)
        self.assertIn("Subtract 3", solution)
        self.assertIn("x = 3", solution)
        
        # Verify solution is correct
        is_correct = self.solver.verify_solution(expression, solution)
        self.assertTrue(is_correct)
    
    def test_simplify_polynomial(self):
        """Test polynomial simplification."""
        expression = "x^2 + 5x + 6"
        solution = self.solver.solve_step_by_step(expression)
        
        # Check solution contains steps
        self.assertIn("Step", solution)
        
        # Verify solution is correct
        is_correct = self.solver.verify_solution(expression, solution)
        self.assertTrue(is_correct)
    
    def test_expand_polynomial(self):
        """Test polynomial expansion."""
        expression = "(2x + 1)(3x + 2)"
        solution = self.solver.solve_step_by_step(expression)
        
        # Check solution contains steps
        self.assertIn("Step", solution)
        
        # Verify solution is correct
        is_correct = self.solver.verify_solution(expression, solution)
        self.assertTrue(is_correct)
    
    def test_simplify_arithmetic(self):
        """Test arithmetic simplification."""
        expression = "2x + 3y - 4x + 5y"
        solution = self.solver.solve_step_by_step(expression)
        
        # Check solution contains steps
        self.assertIn("Step", solution)
        
        # Verify solution is correct
        is_correct = self.solver.verify_solution(expression, solution)
        self.assertTrue(is_correct)
    
    def test_verify_solution(self):
        """Test solution verification."""
        # Test correct solution
        expression = "2x + 3 = 9"
        correct_solution = "Step 1: Subtract 3 from both sides → 2x = 6 Step 2: Divide both sides by 2 → x = 3"
        is_correct = self.solver.verify_solution(expression, correct_solution)
        self.assertTrue(is_correct)
        
        # Test incorrect solution
        incorrect_solution = "Step 1: Add 3 to both sides → 2x = 12 Step 2: Divide both sides by 2 → x = 6"
        is_correct = self.solver.verify_solution(expression, incorrect_solution)
        self.assertFalse(is_correct)
    
    def test_get_solution_steps(self):
        """Test extracting solution steps."""
        expression = "2x + 3 = 9"
        solution = self.solver.solve_step_by_step(expression)
        steps = self.solver.get_solution_steps(expression)
        
        self.assertIsInstance(steps, list)
        self.assertGreater(len(steps), 0)
        for step in steps:
            self.assertIsInstance(step, str)
            self.assertGreater(len(step), 0)
    
    def test_evaluate_expression(self):
        """Test expression evaluation."""
        expression = "2x + 3"
        result = self.solver.evaluate_expression(expression, x_value=2)
        
        # 2*2 + 3 = 7
        self.assertEqual(result, 7.0)

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMathExpressionGenerator))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMathTokenizer))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMathSolver))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 