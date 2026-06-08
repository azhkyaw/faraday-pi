from __future__ import annotations
import typer
from faraday.config import Settings
from faraday.embedder import HttpEmbedder
from faraday.index_store import SqliteVecStore
from faraday.ingest import ingest as run_ingest
from faraday.llm_client import HttpLLMClient
from faraday.rag import RagEngine
from faraday.retriever import Retriever

app = typer.Typer(help="Faraday — air-gapped personal RAG appliance")


@app.command()
def ingest(source: str, db: str = Settings().db_path):
    """Index a folder of documents into the local vector store."""
    s = Settings.from_env()
    store = SqliteVecStore(db, dim=s.embed_dim)
    stats = run_ingest(source, store=store, embedder=HttpEmbedder(s),
                       chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    store.close()
    typer.echo(f"Indexed {stats.documents} docs, {stats.chunks} chunks "
               f"({stats.skipped} skipped).")


@app.command()
def ask(question: str, db: str = Settings().db_path):
    """Answer a question from the indexed documents (fully offline)."""
    s = Settings.from_env()
    store = SqliteVecStore(db, dim=s.embed_dim)
    engine = RagEngine(Retriever(HttpEmbedder(s), store), HttpLLMClient(s),
                       top_k=s.top_k, max_tokens=s.max_tokens)
    ans = engine.answer(question)
    store.close()
    typer.echo("\n" + ans.text + "\n")
    typer.echo("Sources:")
    for i, rc in enumerate(ans.sources, start=1):
        typer.echo(f"  [{i}] {rc.chunk.source} (score {rc.score:.3f})")
    if ans.invalid_citations:
        typer.secho(f"  ! hallucinated citations: {ans.invalid_citations}",
                    fg=typer.colors.RED)


if __name__ == "__main__":
    app()
