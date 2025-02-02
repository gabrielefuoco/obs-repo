# Constants for word limits
MIN_WORDS = 400
MAX_WORDS = 750
MAX_WORDS_2 = 500

def split_text_by_headers(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """Split text into chunks using ## headers as primary split points."""
    words = text.split()
    chunks = []
    
    while words:
        current_chunk = words[:max_words]
        split_found = False
        
        # Look for ## headers or newlines between min_words and max_words
        for i in range(min_words - 1, len(current_chunk)):
            # Check if current word is '##' but not '###' or more
            if current_chunk[i].startswith('##') and not current_chunk[i].startswith('###'):
                chunks.append(' '.join(current_chunk[:i+1]))
                words = words[i+1:]
                split_found = True
                break
            elif current_chunk[i] == '\n':
                chunks.append(' '.join(current_chunk[:i+1]))
                words = words[i+1:]
                split_found = True
                break
        
        if not split_found:
            # If no natural split point found, use sentence-based split
            chunk_text = ' '.join(current_chunk)
            result, remaining = split_by_sentence(chunk_text)
            chunks.append(result)
            words = remaining.split() + words[len(current_chunk):]
    
    return chunks

def split_text_natural(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """Split text into chunks using any headers or newlines as split points."""
    words = text.split()
    chunks = []
    
    while words:
        current_chunk = words[:max_words]
        split_found = False
        
        # Look for any headers or newlines between min_words and max_words
        for i in range(min_words - 1, len(current_chunk)):
            if current_chunk[i] == '\n' or current_chunk[i].startswith('#'):
                chunks.append(' '.join(current_chunk[:i+1]))
                words = words[i+1:]
                split_found = True
                break
        
        if not split_found:
            # If no natural split point found, use sentence-based split
            chunk_text = ' '.join(current_chunk)
            result, remaining = split_by_sentence(chunk_text)
            chunks.append(result)
            words = remaining.split() + words[len(current_chunk):]
    
    return chunks

def split_by_sentence(text, min_words=MIN_WORDS, max_words=MAX_WORDS_2):
    """Split text into chunks by sentence endings."""
    words = text.split()
    for i in range(min_words, min(len(words), max_words)):
        if '.' in words[i]:
            return ' '.join(words[:i+1]), ' '.join(words[i+1:])
    return ' '.join(words[:max_words]), ' '.join(words[max_words:])
