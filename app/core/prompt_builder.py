"""
Prompt builder for RAG chain.
Assembles prompts with memory context and retrieved documents.
"""
from typing import List

from langchain_core.documents import Document
from langsmith import traceable


class PromptBuilder:
    """Build prompts for the RAG chain."""
    
    # Maximum characters for retrieved context to avoid exceeding LLM token limit
    MAX_CONTEXT_CHARS = 6000
    MAX_MEMORY_CHARS = 2000
    
    SYSTEM_PROMPT = """你是一位专业的私人食谱与饮食规划师，擅长根据用户的口味偏好、健康状况和可用食材提供个性化的饮食建议。

请根据以下信息回答用户问题：

## 对话上下文
{memory_context}

## 参考资料
{retrieved_context}

回答要求：
1. 优先基于参考资料中的食谱和营养知识回答
2. 给出具体的食材用量和烹饪步骤
3. 考虑用户的饮食偏好和健康需求
4. 如有多种建议，请列出优缺点
5. 如果参考资料中没有相关信息，请明确说明，并基于你的专业知识给出建议
"""
    
    @traceable(name="Prompt Builder", run_type="chain")
    def build(
        self,
        query: str,
        retrieved_docs: List[Document],
        memory_docs: List[Document] = None
    ) -> str:
        """
        Build the final prompt with all context.
        
        Args:
            query: User's question
            retrieved_docs: Retrieved documents from knowledge base
            memory_docs: Retrieved conversation history
            
        Returns:
            Formatted prompt string
        """
        # Format memory context
        memory_context = self._format_memory_context(memory_docs)
        
        # Format retrieved context
        retrieved_context = self._format_retrieved_context(retrieved_docs)
        
        # Build final prompt
        prompt = self.SYSTEM_PROMPT.format(
            memory_context=memory_context,
            retrieved_context=retrieved_context,
        )
        
        return prompt
    
    def _format_memory_context(self, memory_docs: List[Document] = None) -> str:
        """
        Format conversation history for prompt.
        
        Args:
            memory_docs: List of memory documents
            
        Returns:
            Formatted memory context string
        """
        if not memory_docs:
            return "（无对话历史）"
        
        formatted_parts = []
        total_chars = 0
        for doc in memory_docs:
            role = doc.metadata.get("role", "unknown")
            content = doc.page_content
            
            if role == "user":
                line = f"用户: {content}"
            elif role == "assistant":
                line = f"助手: {content}"
            else:
                continue
            
            if total_chars + len(line) > self.MAX_MEMORY_CHARS:
                break
            formatted_parts.append(line)
            total_chars += len(line)
        
        return "\n".join(formatted_parts) if formatted_parts else "（无对话历史）"
    
    def _format_retrieved_context(self, retrieved_docs: List[Document]) -> str:
        """
        Format retrieved documents for prompt.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string with numbered references
        """
        if not retrieved_docs:
            return "（无相关参考资料）"
        
        formatted_parts = []
        total_chars = 0
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("filename", "未知来源")
            content = doc.page_content
            
            entry = f"[{i}] 来源: {source}\n{content}"
            if total_chars + len(entry) > self.MAX_CONTEXT_CHARS:
                break
            formatted_parts.append(entry)
            total_chars += len(entry)
        
        return "\n\n".join(formatted_parts) if formatted_parts else "（无相关参考资料）"
    
    def format_sources(self, retrieved_docs: List[Document]) -> List[dict]:
        """
        Format retrieved documents as source references for frontend.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            List of source dictionaries
        """
        sources = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = {
                "index": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", ""),
                "filename": doc.metadata.get("filename", "未知来源"),
                "score": doc.metadata.get("rerank_score", doc.metadata.get("rrf_score", 0)),
            }
            sources.append(source)
        
        return sources
