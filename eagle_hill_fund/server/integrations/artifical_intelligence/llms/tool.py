class LLMBaseClass:
    def __init__(self):
        pass

    def split_text(self, text, max_tokens=3000):
        """
        Splits text into smaller chunks that fit within the given LLM token limit.
        """
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_tokens:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        chunks.append(current_chunk.strip())

        return chunks
