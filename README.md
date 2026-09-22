# Multimodal RAG Fusion Pipeline

A notebook-based retrieval-augmented generation (RAG) prototype for asking questions about PDF documents containing text, tables, and images. The pipeline extracts structured content, creates searchable multimodal descriptions, stores embeddings in ChromaDB, and uses an OpenAI model to answer questions from the most relevant source chunks.

The included example indexes the *Attention Is All You Need* paper.

## How it works

1. **Parse a PDF** with Unstructured's high-resolution parser, retaining tables and embedded images.
2. **Chunk by title** so that related sections of a document stay together.
3. **Enrich mixed-content chunks** with GPT-4o summaries when a chunk contains a table or image; text-only chunks are kept as-is.
4. **Embed and persist** the enriched documents with OpenAI embeddings and ChromaDB.
5. **Retrieve and answer** by fetching the most relevant chunks and passing their text, tables, and images to GPT-4o.

## Repository contents

| Path | Purpose |
| --- | --- |
| [`multi_modal_rag.ipynb`](multi_modal_rag.ipynb) | End-to-end ingestion, indexing, retrieval, and answer-generation workflow. |
| [`docs/attention-is-all-you-need.pdf`](docs/attention-is-all-you-need.pdf) | Sample source document. |
| `dbv1/chroma_db/` | Persisted ChromaDB vector store created by the notebook. |

## Prerequisites

- Python 3.11 or later
- An OpenAI API key
- System dependencies required by Unstructured's high-resolution PDF processing (see the [Unstructured installation documentation](https://docs.unstructured.io/open-source/installation/full-installation))

## Setup

Create and activate an isolated Python environment, then install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install \
  jupyter \
  unstructured[pdf] \
  langchain-openai \
  langchain-chroma \
  chromadb \
  python-dotenv
```

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

Start Jupyter and open the notebook:

```bash
jupyter notebook multi_modal_rag.ipynb
```

Run the cells in order. To index another document, change the `file_path` value near the beginning of the notebook. The vector store is written to `dbv1/chroma_db` by default.

## Configuration

The current notebook uses:

- `gpt-4o` for image/table-aware summaries and final answer generation
- `text-embedding-3-small` for vector embeddings
- ChromaDB with cosine similarity and the top three retrieved chunks
- Unstructured's `hi_res` PDF strategy, table inference, and base64 image extraction

These values are defined directly in [`multi_modal_rag.ipynb`](multi_modal_rag.ipynb) and can be adjusted to fit document size, quality, latency, or cost requirements.

## Notes

- Processing PDFs with the `hi_res` strategy can be resource-intensive, especially for image-heavy documents.
- The persisted vector store is a generated artifact. Delete `dbv1/chroma_db` and re-run the indexing cells to rebuild it from a source PDF.
- Keep your OpenAI key private. Do not commit real credentials to `.env`.

## License

No license has been specified for this project. Add one before distributing or reusing the code outside this repository.
