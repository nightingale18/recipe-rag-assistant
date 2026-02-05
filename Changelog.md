# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-01-15

### Added
- Initial release
- Real-time recipe search with FAISS
- Recipe upload and editing interface
- Answer validation system
- Real-time file monitoring
- Support for markdown recipe format
- Filtering by cuisine, diet, time
- Statistics and analytics dashboard

### Features
- Semantic search using sentence transformers
- Auto-indexing of recipe files
- Version tracking for recipes
- Confidence scoring for answers
- Batch operations support
- Export recipes as JSON/markdown

### Technical
- FAISS vector store for fast similarity search
- Sentence transformer embeddings (all-MiniLM-L6-v2)
- Streamlit web interface
- Real-time file system monitoring
- Pickle-based data persistence