def apply_domain_rules(cleaned_text):
    """
    Apply domain-specific rules to the normalized text.
    
    Args:
        cleaned_text (str): Cleaned and normalized text.
    
    Returns:
        str: Text with domain-specific corrections applied.
    """
    rules = {
        "Goo": "Good",  # Example rule
        "somttlidj": "something good",  # Custom correction
    }
    for key, value in rules.items():
        cleaned_text = cleaned_text.replace(key, value)
    
    return cleaned_text

