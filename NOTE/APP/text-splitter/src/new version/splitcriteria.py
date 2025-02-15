import re
from abc import ABC, abstractmethod

class BaseTextSplitter(ABC):
    """Classe base per gli text splitter con funzionalità comuni"""
    
    def __init__(self, min_words=400, max_words=2000):
        self.min_words = min_words
        self.max_words = max_words
        
    def split(self, text):
        """Metodo principale per lo splitting del testo"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for line in lines:
            line_word_count = len(line.split())
            
            if self._should_split(line, current_word_count + line_word_count):
                chunks, current_chunk, current_word_count = self._handle_split(
                    chunks, current_chunk, current_word_count
                )
                
            current_chunk.append(line)
            current_word_count += line_word_count
            
        # Gestione ultimo chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return self._post_process(chunks)
    
    @abstractmethod
    def _should_split(self, line, projected_word_count):
        """Condizione per decidere se splittare (da implementare nelle sottoclassi)"""
        pass
    
    def _handle_split(self, chunks, current_chunk, current_word_count):
        """Gestione comune dello splitting"""
        chunk_text = '\n'.join(current_chunk)
        
        if current_word_count >= self.min_words:
            chunks.append(chunk_text)
            return chunks, [], 0
        
        # Unione chunk piccoli
        if chunks:
            chunks[-1] += '\n' + chunk_text
        else:
            chunks.append(chunk_text)
            
        return chunks, [], 0
    
    def _post_process(self, chunks):
        """Post-processing comune"""
        # Unione chunk piccoli
        merged = []
        accumulator = []
        acc_count = 0
        
        for chunk in chunks:
            chunk_count = len(chunk.split())
            if acc_count + chunk_count < self.min_words:
                accumulator.append(chunk)
                acc_count += chunk_count
            else:
                if accumulator:
                    merged.append('\n\n'.join(accumulator))
                    accumulator = []
                    acc_count = 0
                merged.append(chunk)
                
        if accumulator:
            if merged:
                merged[-1] += '\n\n' + '\n\n'.join(accumulator)
            else:
                merged.append('\n\n'.join(accumulator))
                
        # Emergency split per chunk troppo grandi
        final_chunks = []
        for chunk in merged:
            if len(chunk.split()) <= self.max_words:
                final_chunks.append(chunk)
                continue
                
            final_chunks.extend(self._emergency_split(chunk))
            
        return final_chunks
    
    def _emergency_split(self, chunk):
        """Splitting di emergenza comune"""
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        temp_chunk = []
        temp_count = 0
        result = []
        
        for sentence in sentences:
            words = sentence.split()
            if temp_count + len(words) > self.max_words:
                if temp_count >= self.min_words:
                    result.append(' '.join(temp_chunk))
                    temp_chunk = []
                    temp_count = 0
                else:
                    temp_chunk.extend(words[:self.max_words - temp_count])
                    result.append(' '.join(temp_chunk))
                    temp_chunk = words[self.max_words - temp_count:]
                    temp_count = len(temp_chunk)
            else:
                temp_chunk.extend(words)
                temp_count += len(words)
                
        if temp_chunk:
            result.append(' '.join(temp_chunk))
            
        return result

class MarkdownSplitter(BaseTextSplitter):
    """Splitter per Markdown con gestione header e blocchi di codice"""
    
    def __init__(self, header_levels=(2,), code_block_delimiter='```', **kwargs):
        super().__init__(**kwargs)
        self.header_levels = header_levels
        self.code_block_delimiter = code_block_delimiter
        self.in_code_block = False
        self.last_valid_header = -1
        self.current_chunk = []  # Aggiungi questa linea

    def _should_split(self, line, projected_word_count):
        stripped = line.lstrip()
        
        # Gestione blocchi di codice
        if self.code_block_delimiter and stripped.startswith(self.code_block_delimiter):
            self.in_code_block = not self.in_code_block
            
        if self.in_code_block:
            return False
            
        # Rilevazione header
        if stripped.startswith('#'):
            header_level = 0
            while header_level < len(stripped) and stripped[header_level] == '#':
                header_level += 1
                
            if (header_level in self.header_levels and
                header_level < len(stripped) and
                stripped[header_level] in (' ', '\t')):
                self.last_valid_header = len(self.current_chunk) - 1
                
        return projected_word_count >= self.max_words or self.last_valid_header != -1

class NaturalTextSplitter(BaseTextSplitter):
    """Splitter per testo generico con rilevamento header e paragrafi"""
    
    def __init__(self, header_patterns=None, **kwargs):
        super().__init__(**kwargs)
        self.header_patterns = header_patterns or [
            r'^\s*[A-Z][A-Z\s:]+\s*$',
            r'^\s*\d+\.\s+[A-Z]',
            r'^\s*Chapter\s+\d+',
            r'^\s*SECTION\b',
            r'^\s*[IVX]+\.\s+[A-Z]'
        ]
        self.prev_line_empty = False
        self.current_chunk = []  # Aggiungi questa linea

    def _should_split(self, line, projected_word_count):
        stripped = line.strip()
        is_header = any(re.match(p, stripped) for p in self.header_patterns)
        is_new_para = self.prev_line_empty and stripped
        
        self.prev_line_empty = not stripped
        
        return (projected_word_count >= self.max_words or 
                is_header or 
                is_new_para)