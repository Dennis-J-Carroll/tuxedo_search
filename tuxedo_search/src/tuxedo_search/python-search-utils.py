"""
Search Engine Utilities for Data Import and Management
"""
import pandas as pd
import numpy as np
import json
import csv
import aiofiles
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Generator
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import yaml
from tqdm import tqdm
import multiprocessing as mp
from dataclasses import dataclass
import pickle
from datetime import datetime

@dataclass
class IndexStats:
    """Statistics about indexed documents"""
    total_documents: int
    total_tokens: int
    unique_terms: int
    index_size_mb: float
    indexing_time_ms: float

class DataLoader:
    """Utility for loading and preprocessing data from various sources"""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    async def process_csv(self, 
                         filepath: Union[str, Path],
                         content_col: str,
                         id_col: str,
                         encoding: str = 'utf-8',
                         delimiter: str = ',') -> Generator[Dict, None, None]:
        """Process CSV files in chunks"""
        try:
            for chunk in pd.read_csv(filepath, 
                                   chunksize=self.chunk_size,
                                   encoding=encoding,
                                   delimiter=delimiter):
                for _, row in chunk.iterrows():
                    yield {
                        'id': str(row[id_col]),
                        'content': str(row[content_col])
                    }
        except Exception as e:
            self.logger.error(f"Error processing CSV {filepath}: {str(e)}")
            raise
    
    async def process_jsonl(self, 
                           filepath: Union[str, Path],
                           content_key: str,
                           id_key: str) -> Generator[Dict, None, None]:
        """Process JSONL files line by line"""
        async with aiofiles.open(filepath, mode='r', encoding='utf-8') as f:
            buffer = []
            async for line in f:
                try:
                    data = json.loads(line)
                    doc = {
                        'id': str(data[id_key]),
                        'content': str(data[content_key])
                    }
                    buffer.append(doc)
                    
                    if len(buffer) >= self.chunk_size:
                        yield from buffer
                        buffer = []
                except json.JSONDecodeError:
                    self.logger.warning(f"Skipping invalid JSON line in {filepath}")
                    continue
            
            if buffer:
                yield from buffer

class SearchEngineUtils:
    """Extended utilities for the search engine"""
    
    def __init__(self, search_engine, max_workers: int = None):
        self.engine = search_engine
        self.max_workers = max_workers or mp.cpu_count()
        self.loader = DataLoader()
        self.logger = logging.getLogger(__name__)
    
    async def bulk_index_csv(self,
                            filepath: Union[str, Path],
                            content_col: str,
                            id_col: str,
                            batch_size: int = 1000,
                            **csv_kwargs) -> IndexStats:
        """Bulk index documents from CSV file"""
        start_time = datetime.now()
        stats = IndexStats(0, 0, 0, 0.0, 0.0)
        
        async for batch in self.loader.process_csv(filepath, content_col, id_col, **csv_kwargs):
            await self._index_batch(batch)
            stats.total_documents += len(batch)
        
        stats.indexing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return await self._finalize_stats(stats)
    
    async def bulk_index_jsonl(self,
                              filepath: Union[str, Path],
                              content_key: str,
                              id_key: str,
                              batch_size: int = 1000) -> IndexStats:
        """Bulk index documents from JSONL file"""
        start_time = datetime.now()
        stats = IndexStats(0, 0, 0, 0.0, 0.0)
        
        async for batch in self.loader.process_jsonl(filepath, content_key, id_key):
            await self._index_batch(batch)
            stats.total_documents += len(batch)
        
        stats.indexing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return await self._finalize_stats(stats)
    
    async def bulk_index_directory(self,
                                 directory: Union[str, Path],
                                 file_pattern: str = "**/*",
                                 recursive: bool = True) -> IndexStats:
        """Recursively index all files in a directory"""
        start_time = datetime.now()
        stats = IndexStats(0, 0, 0, 0.0, 0.0)
        directory = Path(directory)
        
        files = list(directory.glob(file_pattern) if recursive else directory.glob(file_pattern))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for file in tqdm(files, desc="Indexing files"):
                try:
                    content = await self._read_file(file)
                    doc = {
                        'id': str(file.relative_to(directory)),
                        'content': content
                    }
                    await self._index_batch([doc])
                    stats.total_documents += 1
                except Exception as e:
                    self.logger.error(f"Error indexing {file}: {str(e)}")
        
        stats.indexing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return await self._finalize_stats(stats)
    
    async def bulk_index_websites(self,
                                urls: List[str],
                                max_depth: int = 2) -> IndexStats:
        """Index content from websites with crawling"""
        start_time = datetime.now()
        stats = IndexStats(0, 0, 0, 0.0, 0.0)
        
        async def crawl_url(url: str, depth: int = 0):
            if depth > max_depth:
                return
            
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract main content
                content = self._extract_main_content(soup)
                
                doc = {
                    'id': url,
                    'content': content,
                    'metadata': {
                        'title': soup.title.string if soup.title else '',
                        'crawled_at': datetime.now().isoformat()
                    }
                }
                
                await self._index_batch([doc])
                stats.total_documents += 1
                
                # Find and crawl links if not at max depth
                if depth < max_depth:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        if link['href'].startswith(url):
                            await crawl_url(link['href'], depth + 1)
                
            except Exception as e:
                self.logger.error(f"Error crawling {url}: {str(e)}")
        
        tasks = [crawl_url(url) for url in urls]
        await asyncio.gather(*tasks)
        
        stats.indexing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return await self._finalize_stats(stats)
    
    async def export_index(self, 
                          output_dir: Union[str, Path],
                          format: str = 'json') -> Path:
        """Export the search index to file"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"search_index_{timestamp}.{format}"
        
        # Get all documents from the engine
        all_docs = await self.engine.get_all_documents()
        
        if format == 'json':
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(json.dumps(all_docs, indent=2))
        elif format == 'pickle':
            with open(output_file, 'wb') as f:
                pickle.dump(all_docs, f)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return output_file
    
    async def _index_batch(self, batch: List[Dict]) -> None:
        """Index a batch of documents with error handling"""
        try:
            async with asyncio.TaskGroup() as tg:
                for doc in batch:
                    tg.create_task(
                        self.engine.index_document(doc['content'], doc['id'])
                    )
        except Exception as e:
            self.logger.error(f"Error indexing batch: {str(e)}")
            raise
    
    @staticmethod
    async def _read_file(filepath: Path) -> str:
        """Read file content with appropriate encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                async with aiofiles.open(filepath, encoding=encoding) as f:
                    return await f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file {filepath} with any supported encoding")
    
    @staticmethod
    def _extract_main_content(soup: BeautifulSoup) -> str:
        """Extract main content from HTML, skipping navigation, headers, etc."""
        # Remove unwanted tags
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        # Find main content (adjust selectors based on common patterns)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        
        if main_content:
            return main_content.get_text(separator=' ', strip=True)
        return soup.get_text(separator=' ', strip=True)
    
    async def _finalize_stats(self, stats: IndexStats) -> IndexStats:
        """Calculate final statistics about the indexed documents"""
        # Get additional stats from the engine
        engine_stats = await self.engine.get_statistics()
        
        stats.unique_terms = engine_stats.get('unique_terms', 0)
        stats.total_tokens = engine_stats.get('total_tokens', 0)
        stats.index_size_mb = engine_stats.get('index_size_bytes', 0) / (1024 * 1024)
        
        return stats

# Example usage
async def main():
    from fastsearch import SearchEngine
    
    # Initialize engine and utilities
    engine = SearchEngine()
    utils = SearchEngineUtils(engine)
    
    # Index from CSV
    csv_stats = await utils.bulk_index_csv(
        'documents.csv',
        content_col='text',
        id_col='doc_id'
    )
    print(f"CSV Indexing Stats: {csv_stats}")
    
    # Index from directory
    dir_stats = await utils.bulk_index_directory(
        'document_directory',
        file_pattern="**/*.txt"
    )
    print(f"Directory Indexing Stats: {dir_stats}")
    
    # Index from websites
    web_stats = await utils.bulk_index_websites([
        'https://example.com',
        'https://example.org'
    ])
    print(f"Website Indexing Stats: {web_stats}")
    
    # Export index
    output_file = await utils.export_index('exports', format='json')
    print(f"Index exported to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())