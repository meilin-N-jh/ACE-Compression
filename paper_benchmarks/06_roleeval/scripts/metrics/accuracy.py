import re
from collections import defaultdict

def extract_answer(response: str) -> str:
    """Extract answer from model response."""
    if not response:
        return ""

    # Prefer explicit answer statements
    explicit_patterns = [
        r"答案\s*[:：]?\s*([A-D])",
        r"正确答案\s*[:：]?\s*([A-D])",
        r"Answer\s*[:：]?\s*([A-D])",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # If response includes thinking, take the last occurrence to avoid option lists
    matches = re.findall(r"[A-D]", response)
    if matches:
        return matches[-1]

    matches = re.findall(r"[a-d]", response)
    if matches:
        return matches[-1].upper()

    # Check for Chinese answer phrases; use the last matched pattern
    chinese_mapping = {
        '选项A': 'A', '答案A': 'A', '是A': 'A',
        '选项B': 'B', '答案B': 'B', '是B': 'B',
        '选项C': 'C', '答案C': 'C', '是C': 'C',
        '选项D': 'D', '答案D': 'D', '是D': 'D',
    }
    last_answer = ""
    for pattern, answer in chinese_mapping.items():
        if pattern in response:
            last_answer = answer
    return last_answer

def calculate_accuracy(results: list, by_category=False, by_split=False) -> dict:
    """Calculate accuracy from results"""
    stats = {
        'total': 0,
        'correct': 0,
        'accuracy': 0.0,
    }
    
    if by_category:
        stats['by_category'] = defaultdict(lambda: {'total': 0, 'correct': 0, 'accuracy': 0.0})
    
    if by_split:
        stats['by_split'] = defaultdict(lambda: {'total': 0, 'correct': 0, 'accuracy': 0.0})
    
    for result in results:
        stats['total'] += 1
        if result.get('is_correct'):
            stats['correct'] += 1
        
        if by_category and 'category' in result:
            cat = result['category']
            stats['by_category'][cat]['total'] += 1
            if result.get('is_correct'):
                stats['by_category'][cat]['correct'] += 1
        
        if by_split and 'split' in result:
            split = result['split']
            stats['by_split'][split]['total'] += 1
            if result.get('is_correct'):
                stats['by_split'][split]['correct'] += 1
    
    # Calculate accuracies
    if stats['total'] > 0:
        stats['accuracy'] = stats['correct'] / stats['total']
    
    if by_category:
        for cat in stats['by_category']:
            total = stats['by_category'][cat]['total']
            if total > 0:
                stats['by_category'][cat]['accuracy'] = stats['by_category'][cat]['correct'] / total
    
    if by_split:
        for split in stats['by_split']:
            total = stats['by_split'][split]['total']
            if total > 0:
                stats['by_split'][split]['accuracy'] = stats['by_split'][split]['correct'] / total
    
    return stats

def print_metrics(stats: dict, model_name: str = ""):
    """Print formatted metrics"""
    if model_name:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
    
    print(f"Overall Accuracy: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    if 'by_category' in stats:
        print(f"\n{'Category':<30} {'Accuracy':<15} {'Count'}")
        print("-" * 60)
        for cat, cat_stats in sorted(stats['by_category'].items()):
            acc = cat_stats['accuracy']
            count = cat_stats['total']
            print(f"{cat:<30} {acc:.4f}            {cat_stats['correct']}/{count}")
    
    if 'by_split' in stats:
        print(f"\n{'Split':<30} {'Accuracy':<15} {'Count'}")
        print("-" * 60)
        for split, split_stats in sorted(stats['by_split'].items()):
            acc = split_stats['accuracy']
            count = split_stats['total']
            print(f"{split:<30} {acc:.4f}            {split_stats['correct']}/{count}")
