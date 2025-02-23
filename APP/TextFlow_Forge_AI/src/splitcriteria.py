import re
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Pattern

class BaseTextSplitter(ABC):
    """Classe base per lo splitting del testo con max_words adattivo."""
    
    def __init__(self, min_words: int = 400, max_words: int = 2000, margin_percent: float = 0.2, text: Optional[str] = None):
        self.min_words = min_words
        self.max_words = max_words
        self.margin = int(max_words * margin_percent)
        
        if text:
            self.max_words = self._compute_adaptive_max_words(text)
            self.margin = int(self.max_words * margin_percent)
    
    def _word_count(self, text: str) -> int:
        return len(text.split())
    
    def _compute_adaptive_max_words(self, text: str) -> int:
        """Calcola un max_words adattivo in base alla lunghezza del testo."""
        word_count = self._word_count(text)
        if word_count <= self.max_words:
            return self.max_words
        
        num_chunks = word_count / self.max_words
        num_chunks = math.ceil(num_chunks) if num_chunks % 1 > 0.3 else math.floor(num_chunks)
        adaptive_max_words = word_count // num_chunks
        return max(adaptive_max_words, int(self.min_words * 1.2))
    
    def _has_header_within(self, text: str, word_limit: int) -> bool:
        """
        Verifica se nel testo sono presenti header (h3 o superiori) entro il limite di parole specificato.
        """
        words = text.split()
        limited_text = " ".join(words[:word_limit])
        # Cerca righe che iniziano con almeno 3 '#' seguiti da uno spazio
        return bool(re.search(r'^\s*#{3,}\s', limited_text, re.MULTILINE))
    
    def _fallback_split(self, text: str) -> List[str]:
        """
        Splitting di fallback: se non vengono riconosciuti header entro 2*max_words,
        il testo viene splittato "normalmente", interrompendo al newline o al punto più vicino al limite.
        """
        words = text.split()
        chunks = []
        start = 0
        total_words = len(words)
        
        while start < total_words:
            end = min(start + self.max_words, total_words)
            candidate = " ".join(words[start:end])
            # Tenta di trovare un punto di rottura: newline o punto
            split_index = candidate.rfind('\n')
            if split_index == -1:
                split_index = candidate.rfind('.')
                if split_index != -1:
                    split_index += 1  # includi il punto
            if split_index != -1:
                pre_split = candidate[:split_index].strip()
                split_word_count = len(pre_split.split())
                if split_word_count == 0:
                    split_word_count = end - start
            else:
                split_word_count = end - start
            
            chunks.append(" ".join(words[start:start+split_word_count]))
            start += split_word_count
        
        return chunks
    
    def split(self, text: str) -> List[str]:
        """
        Splitta il testo in chunk basandosi su header riconosciuti oppure, in assenza
        di header (o se non presenti entro 2*max_words), utilizza il fallback.
        """
        if not self._has_header_within(text, 2 * self.max_words):
            return self._fallback_split(text)
        
        paragraphs = self._split_into_real_paragraphs(text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        deferred_paragraph = None
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            if deferred_paragraph:
                if current_word_count:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [deferred_paragraph]
                current_word_count = self._word_count(deferred_paragraph)
                deferred_paragraph = None
            
            paragraph_word_count = self._word_count(paragraph)
            total_projected_words = current_word_count + paragraph_word_count
            
            if self._is_header_paragraph(paragraph):
                if total_projected_words <= self.max_words + self.margin:
                    current_chunk.append(paragraph)
                    current_word_count = total_projected_words
                else:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                    deferred_paragraph = paragraph
            else:
                if total_projected_words <= self.max_words + self.margin:
                    current_chunk.append(paragraph)
                    current_word_count = total_projected_words
                else:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [paragraph]
                    current_word_count = paragraph_word_count
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        if deferred_paragraph:
            chunks.append(deferred_paragraph)
        
        return self._post_process(chunks)
    
    def _split_into_real_paragraphs(self, text: str) -> List[str]:
        """
        Divide il testo in paragrafi basandosi su qualsiasi livello di header o sul termine del file.
        Mantiene l'integrità dei paragrafi header (h3 e superiori).
        """
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            stripped = line.lstrip()
            is_header = False
            if stripped.startswith('#'):
                header_level = len(stripped) - len(stripped.lstrip('#'))
                if len(stripped) > header_level and stripped[header_level] in (' ', '\t'):
                    # Considera come header solo se è di livello >= 3
                    if header_level >= 3:
                        is_header = True
            if is_header:
                if current_paragraph:
                    paragraphs.append('\n'.join(current_paragraph))
                current_paragraph = [line]
            else:
                current_paragraph.append(line)
        if current_paragraph:
            paragraphs.append('\n'.join(current_paragraph))
        return paragraphs
    
    def _is_header_paragraph(self, paragraph: str) -> bool:
        """
        Verifica se un paragrafo è un header di livello 3 o superiore (h3, h4, h5, ecc.),
        escludendo h1 e h2.
        """
        stripped = paragraph.lstrip()
        if not stripped.startswith('#'):
            return False
        header_level = len(stripped) - len(stripped.lstrip('#'))
        return header_level >= 3 and (len(stripped) > header_level and stripped[header_level] in (' ', '\t'))
    
    def _post_process(self, chunks: List[str]) -> List[str]:
        """Unisce i chunk piccoli per rispettare il minimo di parole."""
        merged = []
        accumulator = []
        acc_count = 0
        
        for chunk in chunks:
            chunk_count = self._word_count(chunk)
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
                
        return merged
    
    @abstractmethod
    def _should_split(self, line: str, projected_word_count: int) -> bool:
        """Condizione per decidere se splittare (da implementare nelle sottoclassi)"""
        pass

class MarkdownSplitter(BaseTextSplitter):
    """Splitter per Markdown con gestione avanzata degli header."""
    
    def __init__(self, 
                 header_levels: Tuple[int, ...] = (1, 2), 
                 custom_header_regex: Optional[Pattern] = None,
                 code_block_delimiter: str = '```', 
                 **kwargs):
        super().__init__(**kwargs)
        self.header_levels = header_levels
        self.custom_header_regex = custom_header_regex
        self.code_block_delimiter = code_block_delimiter
        self.in_code_block = False
        self.header_pattern = re.compile(r'^#+\s')
    
    def _should_split(self, line: str, projected_word_count: int) -> bool:
        stripped = line.lstrip()
        if self.code_block_delimiter and stripped.startswith(self.code_block_delimiter):
            self.in_code_block = not self.in_code_block
        if self.in_code_block:
            return False
        return projected_word_count >= self.max_words + self.margin or self._is_header(line)
    
    def _is_header(self, line: str) -> bool:
        """
        Verifica se una linea è un header usando una regex custom (se fornita)
        o il pattern standard.
        """
        stripped = line.lstrip()
        if self.custom_header_regex and self.custom_header_regex.match(stripped):
            return True
        if self.header_pattern.match(stripped):
            header_level = len(stripped) - len(stripped.lstrip('#'))
            return header_level in self.header_levels and len(stripped) > header_level and stripped[header_level] in (' ', '\t')
        return False

class NaturalTextSplitter(BaseTextSplitter):
    """Splitter per testo generico con rilevamento contestuale."""
    
    def __init__(self, header_patterns: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.header_patterns = header_patterns or [
            r'^\s*[A-Z][A-Z\s:]+\s*$',
            r'^\s*\d+\.\s+[A-Z]',
            r'^\s*Chapter\s+\d+',
            r'^\s*SECTION\b',
            r'^\s*[IVX]+\.\s+[A-Z]'
        ]
        self.prev_line_empty = False
    
    def _should_split(self, line: str, projected_word_count: int) -> bool:
        stripped = line.strip()
        is_header = any(re.match(p, stripped) for p in self.header_patterns)
        is_new_para = self.prev_line_empty and bool(stripped)
        self.prev_line_empty = not bool(stripped)
        return projected_word_count >= self.max_words + self.margin or is_header or is_new_para

