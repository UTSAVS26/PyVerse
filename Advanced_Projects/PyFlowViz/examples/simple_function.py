def calculate_grade(score):
    """Calculate letter grade based on numerical score."""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

def process_student_data(student_scores):
    """Process student scores and return summary."""
    results = []
    
    for student, score in student_scores.items():
        grade = calculate_grade(score)
        results.append({
            'student': student,
            'score': score,
            'grade': grade
        })
    
    return results

# Example usage
if __name__ == "__main__":
    scores = {
        'Alice': 95,
        'Bob': 87,
        'Charlie': 72,
        'Diana': 55
    }
    
    results = process_student_data(scores)
    for result in results:
        print(f"{result['student']}: {result['score']} -> {result['grade']}") 