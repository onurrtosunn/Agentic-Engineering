import os
import shutil
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv(override=True)

# Web/API
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Data and docs
import pandas as pd
from pypdf import PdfReader

# LangChain / LLM / VectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# Constants and default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_MANUALS_DIR = os.path.join(DEFAULT_DATA_DIR, "manuals")
DEFAULT_INDEX_DIR = os.path.join(BASE_DIR, ".shopadvisor_index")

DEFAULT_MODEL = os.getenv("SHOPADVISOR_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Globals (lazy-initialized)
_embeddings = None
_vectorstore: Optional[FAISS] = None
_retriever = None
_qa_chain: Optional[RetrievalQA] = None


class IngestResponse(BaseModel):
	ok: bool
	chunks_indexed: int
	index_dir: str
	message: Optional[str] = None


class AskRequest(BaseModel):
	question: str
	top_k: int = 4


class AskResponse(BaseModel):
	ok: bool
	answer: str
	sources: List[Dict[str, Any]]
	message: Optional[str] = None


def ensure_dirs() -> None:
	os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
	os.makedirs(DEFAULT_MANUALS_DIR, exist_ok=True)
	os.makedirs(DEFAULT_INDEX_DIR, exist_ok=True)


def get_embeddings():
	"""Return embeddings instance. Use OpenAI if key is set; else fallback to local HF model."""
	global _embeddings
	if _embeddings is not None:
		return _embeddings

	preferred = os.getenv("SHOPADVISOR_EMBEDDINGS", "auto").lower()
	if preferred == "openai" or (preferred == "auto" and OPENAI_API_KEY):
		_embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
		return _embeddings

	# Fallback to HuggingFace embeddings (anonymous to avoid expired-token issues)
	model_name = os.getenv("SHOPADVISOR_HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
	# Ensure any existing HF tokens in env do not interfere
	for var in ["HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"]:
		os.environ.pop(var, None)
	_embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"token": None})
	return _embeddings


def load_pdfs_from_directory(directory_path: str) -> List[Document]:
	documents: List[Document] = []
	if not os.path.isdir(directory_path):
		return documents
	for filename in os.listdir(directory_path):
		if not filename.lower().endswith(".pdf"):
			continue
		file_path = os.path.join(directory_path, filename)
		try:
			reader = PdfReader(file_path)
			text_parts: List[str] = []
			for page in reader.pages:
				page_text = page.extract_text() or ""
				if page_text:
					text_parts.append(page_text)
			full_text = "\n".join(text_parts)
			if full_text.strip():
				documents.append(
					Document(
						page_content=full_text,
						metadata={
							"source": file_path,
							"type": "manual",
							"filename": filename,
						},
					)
				)
		except Exception as exc:
			print(f"Failed to read PDF {file_path}: {exc}", flush=True)
	return documents


def load_products_from_csv(csv_path: str) -> List[Document]:
	documents: List[Document] = []
	if not os.path.isfile(csv_path):
		return documents
	try:
		df = pd.read_csv(csv_path)
		for _, row in df.iterrows():
			text_fields: List[str] = []
			for col in [
				"title",
				"name",
				"description",
				"specs",
				"features",
				"review",
				"reviews",
			]:
				val = row.get(col, None)
				if pd.notna(val):
					text_fields.append(str(val))
			combined_text = "\n\n".join(text_fields).strip()
			if not combined_text:
				continue

			metadata: Dict[str, Any] = {"type": "catalog", "source": csv_path}
			for meta_col in ["id", "product_id", "sku", "url", "product_url", "image", "image_url"]:
				val = row.get(meta_col, None)
				if pd.notna(val):
					metadata[meta_col] = str(val)

			documents.append(Document(page_content=combined_text, metadata=metadata))
	except Exception as exc:
		print(f"Failed to read CSV {csv_path}: {exc}", flush=True)
	return documents


def split_documents(documents: List[Document]) -> List[Document]:
	if not documents:
		return []
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=1000, chunk_overlap=150, length_function=len, separators=["\n\n", "\n", " ", ""]
	)
	return text_splitter.split_documents(documents)


def build_or_load_vectorstore(index_dir: str = DEFAULT_INDEX_DIR) -> Optional[FAISS]:
	global _vectorstore
	if _vectorstore is not None:
		return _vectorstore
	embeddings = get_embeddings()
	if os.path.isdir(index_dir) and any(os.scandir(index_dir)):
		_vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
	else:
		_vectorstore = None
	return _vectorstore


def save_vectorstore(vs: FAISS, index_dir: str = DEFAULT_INDEX_DIR) -> None:
	os.makedirs(index_dir, exist_ok=True)
	vs.save_local(index_dir)


def reset_index(index_dir: str = DEFAULT_INDEX_DIR) -> None:
	if os.path.isdir(index_dir):
		shutil.rmtree(index_dir)
	os.makedirs(index_dir, exist_ok=True)
	global _vectorstore, _retriever, _qa_chain
	_vectorstore = None
	_retriever = None
	_qa_chain = None


def ingest_sources(
	products_csv_path: Optional[str] = None,
	manuals_dir: Optional[str] = None,
	reset: bool = False,
	index_dir: str = DEFAULT_INDEX_DIR,
) -> int:
	ensure_dirs()
	if reset:
		reset_index(index_dir)

	products_csv_path = products_csv_path or os.path.join(DEFAULT_DATA_DIR, "products.csv")
	manuals_dir = manuals_dir or DEFAULT_MANUALS_DIR

	documents: List[Document] = []
	documents.extend(load_products_from_csv(products_csv_path))
	documents.extend(load_pdfs_from_directory(manuals_dir))

	chunks = split_documents(documents)
	if not chunks:
		return 0

	embeddings = get_embeddings()
	current_vs = build_or_load_vectorstore(index_dir)
	if current_vs is None:
		vs = FAISS.from_documents(chunks, embeddings)
	else:
		vs = current_vs
		vs.add_documents(chunks)

	save_vectorstore(vs, index_dir)
	global _vectorstore
	_vectorstore = vs
	return len(chunks)


def get_qa_chain(index_dir: str = DEFAULT_INDEX_DIR) -> RetrievalQA:
	global _qa_chain, _retriever
	if _qa_chain is not None:
		return _qa_chain

	vs = build_or_load_vectorstore(index_dir)
	if vs is None:
		raise RuntimeError("Vector index is empty. Run ingestion first.")

	_retriever = vs.as_retriever(search_kwargs={"k": 4})
	llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.0)
	_qa_chain = RetrievalQA.from_chain_type(
		llm=llm,
		retriever=_retriever,
		return_source_documents=True,
	)
	return _qa_chain


def answer_question(question: str, top_k: int = 4) -> AskResponse:
	"""Answer using QA chain if available; fallback to retrieve-only summary if LLM fails."""
	try:
		qa = get_qa_chain()
		result = qa({"query": question})
		answer: str = result.get("result", "")
		source_docs: List[Document] = result.get("source_documents", [])
		sources_payload: List[Dict[str, Any]] = []
		for doc in source_docs:
			sources_payload.append({"snippet": doc.page_content[:4000], "metadata": dict(doc.metadata or {})})
		return AskResponse(ok=True, answer=answer, sources=sources_payload)
	except Exception as e:
		# Retrieval-only fallback
		vs = build_or_load_vectorstore()
		if vs is None:
			return AskResponse(ok=False, answer="", sources=[], message=f"Index empty: {e}")
		docs = vs.similarity_search(question, k=top_k)
		context = "\n\n".join([d.page_content for d in docs])
		sources_payload = [{"snippet": d.page_content[:4000], "metadata": dict(d.metadata or {})} for d in docs]
		return AskResponse(
			ok=True,
			answer=(context[:4000] or "No matching content found."),
			sources=sources_payload,
			message="LLM unavailable; returned top-matching context.",
		)


# FastAPI app (optional server mode)
app = FastAPI(title="ShopAdvisor API", version="0.2.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
	return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(
	products_csv_path: Optional[str] = Form(default=None),
	manuals_dir: Optional[str] = Form(default=None),
	reset: bool = Form(default=False),
) -> IngestResponse:
	try:
		count = ingest_sources(products_csv_path=products_csv_path, manuals_dir=manuals_dir, reset=reset)
		return IngestResponse(ok=True, chunks_indexed=count, index_dir=DEFAULT_INDEX_DIR)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest) -> AskResponse:
	try:
		return answer_question(payload.question, top_k=payload.top_k)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload/csv", response_model=IngestResponse)
def upload_csv(file: UploadFile = File(...)) -> IngestResponse:
	ensure_dirs()
	csv_path = os.path.join(DEFAULT_DATA_DIR, "products.csv")
	with open(csv_path, "wb") as f:
		f.write(file.file.read())
	count = ingest_sources(products_csv_path=csv_path, manuals_dir=None, reset=False)
	return IngestResponse(ok=True, chunks_indexed=count, index_dir=DEFAULT_INDEX_DIR)


@app.post("/upload/manual", response_model=IngestResponse)
def upload_manual(file: UploadFile = File(...)) -> IngestResponse:
	ensure_dirs()
	manual_path = os.path.join(DEFAULT_MANUALS_DIR, file.filename)
	with open(manual_path, "wb") as f:
		f.write(file.file.read())
	count = ingest_sources(products_csv_path=None, manuals_dir=DEFAULT_MANUALS_DIR, reset=False)
	return IngestResponse(ok=True, chunks_indexed=count, index_dir=DEFAULT_INDEX_DIR)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="ShopAdvisor CLI (no FastAPI)")
	sub = parser.add_subparsers(dest="cmd")

	p_ing = sub.add_parser("ingest", help="Ingest CSV/PDF into the vector index")
	p_ing.add_argument("--csv", dest="csv_path", default=None)
	p_ing.add_argument("--manuals", dest="manuals_dir", default=None)
	p_ing.add_argument("--reset", action="store_true")

	p_ask = sub.add_parser("ask", help="Ask a question against the index")
	p_ask.add_argument("question", type=str)
	p_ask.add_argument("--top_k", type=int, default=4)

	p_chat = sub.add_parser("chat", help="Interactive Q&A loop")

	p_srv = sub.add_parser("server", help="Run FastAPI server (optional)")
	p_srv.add_argument("--host", default="0.0.0.0")
	p_srv.add_argument("--port", type=int, default=8000)

	args = parser.parse_args()
	ensure_dirs()

	if args.cmd == "ingest":
		count = ingest_sources(products_csv_path=args.csv_path, manuals_dir=args.manuals_dir, reset=args.reset)
		print({"ok": True, "chunks_indexed": count, "index_dir": DEFAULT_INDEX_DIR})
	elif args.cmd == "ask":
		resp = answer_question(args.question, top_k=args.top_k)
		print({"ok": resp.ok, "answer": resp.answer, "sources": resp.sources, "message": resp.message})
	elif args.cmd == "chat":
		print("Type your question (or 'exit' to quit).")
		while True:
			q = input("Q: ").strip()
			if not q or q.lower() in {"exit", "quit"}:
				break
			resp = answer_question(q)
			print("A:", resp.answer)
			if resp.message:
				print("Note:", resp.message)
			print("Sources:")
			for s in resp.sources:
				meta = s.get("metadata", {})
				print("-", meta.get("filename") or meta.get("url") or meta.get("source"))
	elif args.cmd == "server":
		import uvicorn
		print("Starting ShopAdvisor API on http://%s:%d" % (args.host, args.port), flush=True)
		uvicorn.run("1_foundations.project1:app", host=args.host, port=args.port, reload=False)
	else:
		parser.print_help()
