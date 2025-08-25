"""
Sample Code for CodeSage Analysis

This file contains various functions with different complexity levels
to demonstrate the AI-enhanced code complexity analysis capabilities.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta


@dataclass
class User:
    """Simple user data class."""
    id: int
    name: str
    email: str
    age: int


def simple_function():
    """A simple function with low complexity."""
    return "Hello, World!"


def calculate_average(numbers: List[float]) -> float:
    """Calculate average with basic complexity."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def validate_user_input(user_data: Dict) -> Tuple[bool, List[str]]:
    """Validate user input with moderate complexity."""
    errors = []
    
    if not user_data.get('name'):
        errors.append("Name is required")
    
    if not user_data.get('email'):
        errors.append("Email is required")
    elif '@' not in user_data['email']:
        errors.append("Invalid email format")
    
    if user_data.get('age'):
        try:
            age = int(user_data['age'])
            if age < 0 or age > 150:
                errors.append("Age must be between 0 and 150")
        except ValueError:
            errors.append("Age must be a number")
    
    return len(errors) == 0, errors


def process_user_data(users: List[Dict]) -> List[User]:
    """Process user data with moderate complexity."""
    processed_users = []
    
    for user_data in users:
        is_valid, errors = validate_user_input(user_data)
        
        if is_valid:
            user = User(
                id=user_data.get('id', 0),
                name=user_data['name'],
                email=user_data['email'],
                age=user_data.get('age', 0)
            )
            processed_users.append(user)
        else:
            print(f"Invalid user data: {errors}")
    
    return processed_users


def complex_data_analysis(data: List[Dict], filters: Dict, sort_by: str, limit: int) -> List[Dict]:
    """Complex data analysis function with high complexity."""
    # Apply filters
    filtered_data = []
    for item in data:
        should_include = True
        
        for key, value in filters.items():
            if key in item:
                if isinstance(value, (list, tuple)):
                    if item[key] not in value:
                        should_include = False
                        break
                elif isinstance(value, dict):
                    if 'min' in value and item[key] < value['min']:
                        should_include = False
                        break
                    if 'max' in value and item[key] > value['max']:
                        should_include = False
                        break
                else:
                    if item[key] != value:
                        should_include = False
                        break
            else:
                should_include = False
                break
        
        if should_include:
            filtered_data.append(item)
    
    # Sort data
    if sort_by in filtered_data[0] if filtered_data else False:
        reverse = sort_by.startswith('-')
        sort_key = sort_by[1:] if reverse else sort_by
        
        filtered_data.sort(
            key=lambda x: x.get(sort_key, 0),
            reverse=reverse
        )
    
    # Apply limit
    if limit > 0:
        filtered_data = filtered_data[:limit]
    
    return filtered_data


def very_complex_algorithm(input_data: List[Dict], config: Dict) -> Dict:
    """Very complex algorithm with multiple nested conditions and loops."""
    result = {
        'processed_items': 0,
        'errors': [],
        'warnings': [],
        'statistics': {},
        'output': []
    }
    
    # Validate configuration
    required_fields = ['threshold', 'max_iterations', 'tolerance']
    for field in required_fields:
        if field not in config:
            result['errors'].append(f"Missing required config field: {field}")
            return result
    
    # Process input data
    for i, item in enumerate(input_data):
        try:
            # Complex nested processing
            if item.get('type') == 'numeric':
                value = float(item.get('value', 0))
                
                if value > config['threshold']:
                    # High value processing
                    processed_value = value * 2
                    
                    if processed_value > 1000:
                        # Very high value processing
                        for j in range(min(int(processed_value / 100), config['max_iterations'])):
                            if j % 2 == 0:
                                processed_value += j
                            else:
                                processed_value -= j / 2
                        
                        # Check convergence
                        if abs(processed_value - value) < config['tolerance']:
                            result['warnings'].append(f"Convergence reached at iteration {j}")
                    
                    result['output'].append({
                        'index': i,
                        'original_value': value,
                        'processed_value': processed_value,
                        'iterations': j if 'j' in locals() else 0
                    })
                
                elif value < -config['threshold']:
                    # Negative value processing
                    processed_value = abs(value)
                    
                    # Complex negative value logic
                    if processed_value > 500:
                        for k in range(int(processed_value / 50)):
                            if k % 3 == 0:
                                processed_value *= 0.9
                            elif k % 3 == 1:
                                processed_value *= 1.1
                            else:
                                processed_value = processed_value ** 0.5
                    
                    result['output'].append({
                        'index': i,
                        'original_value': value,
                        'processed_value': processed_value,
                        'negative_processing': True
                    })
                
                else:
                    # Normal value processing
                    result['output'].append({
                        'index': i,
                        'original_value': value,
                        'processed_value': value,
                        'normal_processing': True
                    })
            
            elif item.get('type') == 'text':
                # Text processing
                text = str(item.get('value', ''))
                
                if len(text) > 100:
                    # Long text processing
                    words = text.split()
                    word_count = len(words)
                    
                    if word_count > 20:
                        # Very long text processing
                        processed_text = ' '.join(words[:10]) + '...'
                        
                        # Additional text analysis
                        char_count = len(text)
                        avg_word_length = char_count / word_count if word_count > 0 else 0
                        
                        if avg_word_length > 8:
                            processed_text = processed_text.upper()
                        elif avg_word_length < 4:
                            processed_text = processed_text.lower()
                        
                        result['output'].append({
                            'index': i,
                            'original_text': text[:50] + '...',
                            'processed_text': processed_text,
                            'word_count': word_count,
                            'avg_word_length': avg_word_length
                        })
                    else:
                        result['output'].append({
                            'index': i,
                            'original_text': text,
                            'processed_text': text,
                            'word_count': word_count
                        })
                else:
                    result['output'].append({
                        'index': i,
                        'original_text': text,
                        'processed_text': text,
                        'short_text': True
                    })
            
            else:
                # Unknown type processing
                result['warnings'].append(f"Unknown item type at index {i}")
                result['output'].append({
                    'index': i,
                    'original_value': item.get('value'),
                    'processed_value': None,
                    'error': 'Unknown type'
                })
            
            result['processed_items'] += 1
            
        except Exception as e:
            result['errors'].append(f"Error processing item {i}: {str(e)}")
    
    # Calculate statistics
    if result['output']:
        numeric_values = [
            item['processed_value'] 
            for item in result['output'] 
            if isinstance(item.get('processed_value'), (int, float))
        ]
        
        if numeric_values:
            result['statistics'] = {
                'count': len(numeric_values),
                'sum': sum(numeric_values),
                'average': sum(numeric_values) / len(numeric_values),
                'min': min(numeric_values),
                'max': max(numeric_values)
            }
    
    return result


def deeply_nested_function(data: List[Dict], conditions: List[Dict]) -> List[Dict]:
    """Function with deep nesting to demonstrate nesting depth analysis."""
    result = []
    
    for item in data:
        if item.get('active'):
            if item.get('type') == 'user':
                if item.get('role') == 'admin':
                    if item.get('permissions'):
                        if 'read' in item['permissions']:
                            if 'write' in item['permissions']:
                                if 'delete' in item['permissions']:
                                    if item.get('last_login'):
                                        if isinstance(item['last_login'], str):
                                            try:
                                                login_date = datetime.fromisoformat(item['last_login'])
                                                if login_date > datetime.now() - timedelta(days=30):
                                                    if item.get('email_verified'):
                                                        if item.get('two_factor_enabled'):
                                                            result.append({
                                                                'id': item['id'],
                                                                'status': 'fully_verified_admin',
                                                                'last_login': item['last_login']
                                                            })
                                                        else:
                                                            result.append({
                                                                'id': item['id'],
                                                                'status': 'admin_no_2fa',
                                                                'last_login': item['last_login']
                                                            })
                                                    else:
                                                        result.append({
                                                            'id': item['id'],
                                                            'status': 'admin_email_not_verified',
                                                            'last_login': item['last_login']
                                                        })
                                                else:
                                                    result.append({
                                                        'id': item['id'],
                                                        'status': 'admin_inactive',
                                                        'last_login': item['last_login']
                                                    })
                                            except ValueError:
                                                result.append({
                                                    'id': item['id'],
                                                    'status': 'admin_invalid_date',
                                                    'last_login': item['last_login']
                                                })
                                        else:
                                            result.append({
                                                'id': item['id'],
                                                'status': 'admin_no_login_date'
                                            })
                                    else:
                                        result.append({
                                            'id': item['id'],
                                            'status': 'admin_no_permissions'
                                        })
                                else:
                                    result.append({
                                        'id': item['id'],
                                        'status': 'admin_no_delete_permission'
                                    })
                            else:
                                result.append({
                                    'id': item['id'],
                                    'status': 'admin_no_write_permission'
                                })
                        else:
                            result.append({
                                'id': item['id'],
                                'status': 'admin_no_read_permission'
                            })
                    else:
                        result.append({
                            'id': item['id'],
                            'status': 'admin_no_permissions'
                        })
                else:
                    result.append({
                        'id': item['id'],
                        'status': 'regular_user'
                    })
            else:
                result.append({
                    'id': item['id'],
                    'status': 'non_user_item'
                })
        else:
            result.append({
                'id': item['id'],
                'status': 'inactive'
            })
    
    return result


def function_with_many_parameters(
    user_id: int,
    username: str,
    email: str,
    password: str,
    first_name: str,
    last_name: str,
    age: int,
    phone: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    country: str,
    preferences: Dict,
    settings: Dict,
    metadata: Dict
) -> Dict:
    """Function with many parameters to demonstrate parameter count analysis."""
    # This function has 16 parameters, which is above the recommended threshold
    
    user_data = {
        'id': user_id,
        'username': username,
        'email': email,
        'password': password,  # In real code, this should be hashed
        'profile': {
            'first_name': first_name,
            'last_name': last_name,
            'age': age,
            'phone': phone,
            'address': address,
            'city': city,
            'state': state,
            'zip_code': zip_code,
            'country': country
        },
        'preferences': preferences,
        'settings': settings,
        'metadata': metadata
    }
    
    return user_data


def long_function_with_many_operations():
    """A very long function to demonstrate function length analysis."""
    # This function is intentionally long to demonstrate length analysis
    
    # Initialize variables
    data = []
    processed_data = []
    errors = []
    warnings = []
    statistics = {}
    
    # Generate sample data
    for i in range(100):
        item = {
            'id': i,
            'name': f'Item {i}',
            'value': i * 1.5,
            'category': 'A' if i % 3 == 0 else 'B' if i % 3 == 1 else 'C',
            'active': i % 2 == 0,
            'priority': i % 5 + 1,
            'created_at': datetime.now() - timedelta(days=i),
            'tags': [f'tag{j}' for j in range(i % 4 + 1)],
            'metadata': {
                'source': f'source_{i % 3}',
                'version': f'1.{i % 10}',
                'checksum': f'checksum_{i}'
            }
        }
        data.append(item)
    
    # Process data step 1: Validation
    for item in data:
        if not item.get('name'):
            errors.append(f"Item {item['id']} has no name")
            continue
        
        if item.get('value') < 0:
            warnings.append(f"Item {item['id']} has negative value")
        
        if len(item.get('tags', [])) > 5:
            warnings.append(f"Item {item['id']} has too many tags")
    
    # Process data step 2: Transformation
    for item in data:
        if item.get('active'):
            # Transform active items
            transformed_item = {
                'id': item['id'],
                'name': item['name'].upper(),
                'value': item['value'] * 1.1,  # 10% increase
                'category': item['category'],
                'priority': min(item['priority'] + 1, 5),
                'processed_at': datetime.now(),
                'tags': [tag.upper() for tag in item.get('tags', [])],
                'metadata': item.get('metadata', {})
            }
            
            # Additional processing for high priority items
            if transformed_item['priority'] >= 4:
                transformed_item['urgent'] = True
                transformed_item['value'] *= 1.2  # Additional 20% increase
                
                # Special handling for category A high priority items
                if transformed_item['category'] == 'A':
                    transformed_item['special_handling'] = True
                    transformed_item['value'] *= 1.5  # Additional 50% increase
                    
                    # Add special tags
                    transformed_item['tags'].extend(['urgent', 'special'])
            
            processed_data.append(transformed_item)
        else:
            # Transform inactive items
            transformed_item = {
                'id': item['id'],
                'name': item['name'].lower(),
                'value': item['value'] * 0.9,  # 10% decrease
                'category': item['category'],
                'priority': max(item['priority'] - 1, 1),
                'processed_at': datetime.now(),
                'tags': [tag.lower() for tag in item.get('tags', [])],
                'metadata': item.get('metadata', {})
            }
            
            # Additional processing for low priority items
            if transformed_item['priority'] <= 2:
                transformed_item['low_priority'] = True
                transformed_item['value'] *= 0.8  # Additional 20% decrease
                
                # Special handling for category C low priority items
                if transformed_item['category'] == 'C':
                    transformed_item['deprecated'] = True
                    transformed_item['value'] *= 0.5  # Additional 50% decrease
                    
                    # Add deprecation tags
                    transformed_item['tags'].extend(['deprecated', 'low_value'])
            
            processed_data.append(transformed_item)
    
    # Process data step 3: Aggregation
    category_stats = {}
    priority_stats = {}
    
    for item in processed_data:
        # Category statistics
        category = item['category']
        if category not in category_stats:
            category_stats[category] = {
                'count': 0,
                'total_value': 0,
                'avg_value': 0,
                'urgent_count': 0,
                'special_count': 0
            }
        
        category_stats[category]['count'] += 1
        category_stats[category]['total_value'] += item['value']
        
        if item.get('urgent'):
            category_stats[category]['urgent_count'] += 1
        
        if item.get('special_handling'):
            category_stats[category]['special_count'] += 1
        
        # Priority statistics
        priority = item['priority']
        if priority not in priority_stats:
            priority_stats[priority] = {
                'count': 0,
                'total_value': 0,
                'avg_value': 0
            }
        
        priority_stats[priority]['count'] += 1
        priority_stats[priority]['total_value'] += item['value']
    
    # Calculate averages
    for category in category_stats:
        if category_stats[category]['count'] > 0:
            category_stats[category]['avg_value'] = (
                category_stats[category]['total_value'] / 
                category_stats[category]['count']
            )
    
    for priority in priority_stats:
        if priority_stats[priority]['count'] > 0:
            priority_stats[priority]['avg_value'] = (
                priority_stats[priority]['total_value'] / 
                priority_stats[priority]['count']
            )
    
    # Process data step 4: Final statistics
    statistics = {
        'total_items': len(processed_data),
        'total_value': sum(item['value'] for item in processed_data),
        'avg_value': sum(item['value'] for item in processed_data) / len(processed_data) if processed_data else 0,
        'urgent_items': sum(1 for item in processed_data if item.get('urgent')),
        'special_items': sum(1 for item in processed_data if item.get('special_handling')),
        'low_priority_items': sum(1 for item in processed_data if item.get('low_priority')),
        'deprecated_items': sum(1 for item in processed_data if item.get('deprecated')),
        'category_stats': category_stats,
        'priority_stats': priority_stats,
        'error_count': len(errors),
        'warning_count': len(warnings)
    }
    
    # Process data step 5: Final result
    result = {
        'processed_items': processed_data,
        'statistics': statistics,
        'errors': errors,
        'warnings': warnings,
        'processing_time': datetime.now(),
        'status': 'completed'
    }
    
    return result


def main():
    """Main function to demonstrate the sample code."""
    print("CodeSage Sample Code Analysis")
    print("=" * 40)
    
    # Test simple functions
    print(f"Simple function result: {simple_function()}")
    print(f"Average calculation: {calculate_average([1, 2, 3, 4, 5])}")
    
    # Test user validation
    user_data = {
        'name': 'John Doe',
        'email': 'john@example.com',
        'age': '30'
    }
    is_valid, errors = validate_user_input(user_data)
    print(f"User validation: {is_valid}, Errors: {errors}")
    
    # Test data processing
    users = [user_data]
    processed_users = process_user_data(users)
    print(f"Processed users: {len(processed_users)}")
    
    print("\nSample code analysis complete!")


if __name__ == "__main__":
    main()
