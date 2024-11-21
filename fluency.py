import language_tool_python

def grammar_check(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)

# Example usage
text = "I am very excited to work in your company."
errors = grammar_check(text)
print(f"Grammar errors: {errors}")
