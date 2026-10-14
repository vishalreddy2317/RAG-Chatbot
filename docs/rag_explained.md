# Retrieval-Augmented Generation in 5 minutes

## The problem RAG solves

A language model only knows what was in its training data and what fits in its prompt. Ask it about *your* PDF and it will either refuse or, worse, invent a plausible answer. Fine-tuning a model on every new document is slow and expensive, and it still cannot tell you *where* an answer came from.

## The idea

Instead of making the model memorise the documents, give it the relevant pages at question time:

1. **Index**: split documents into chunks, turn each chunk into a vector (an *embedding*) that captures its meaning, and store the vectors in a vector index.
2. **Retrieve**: turn the question into a vector with the same model and find the chunks whose vectors are closest (cosine similarity).
3. **Generate**: hand those chunks to the answer generator with the instruction "answer only from this context" and cite each chunk's source.

The model's job shrinks from "know everything" to "read these three paragraphs and answer", which it is good at. Because the chunks carry metadata (document, page), every answer can be verified.

## How this project implements each step

| Step | This project |
| --- | --- |
| Parse | PyMuPDF extracts text page by page and flags scanned/encrypted/empty PDFs |
| Chunk | ~800-character, sentence-aligned chunks with 150-character overlap so a fact split across a boundary still appears whole in one chunk; chunks never cross pages |
| Embed | `all-MiniLM-L6-v2` — a free 22 M-parameter Sentence Transformers model, 384-d vectors, runs on CPU |
| Store/search | FAISS exact inner-product index over normalised vectors (= cosine similarity) |
| Decide | A hybrid evidence gate abstains when nothing relevant was retrieved |
| Generate | Extractive sentence ranking by default; optional local (Ollama) or hosted LLM with a grounded prompt |
| Verify | Sentence-level grounding check + page citations with highlighted excerpts |

## Why "extractive" counts as RAG

The *retrieval* half is identical whichever generator is used. The extractive generator simply replaces the LLM with sentence-level semantic ranking: it returns the sentences from the retrieved passages that best match the question. It cannot paraphrase or combine facts, but it also cannot hallucinate, needs no API key, and answers in ~0.1 s on a laptop. The LLM modes add fluency and synthesis on top of the same evidence.

## Where RAG breaks (and what this project does about it)

* **Bad retrieval → bad answer.** If the right chunk is not retrieved, no generator can fix it. The evaluation script measures Recall@k so retrieval quality is visible, and the UI shows similarity scores.
* **Topical-but-wrong passages.** Embeddings capture topic more than answerability. The evidence gate adds a keyword check and the extractive mode reports "no direct answer" when sentence scores are weak.
* **Hallucination in LLM modes.** The prompt forbids outside knowledge, and the grounding check flags unsupported sentences.
* **Scanned PDFs.** No text layer means nothing to embed; the loader detects this and points the user to OCR.
* **Follow-ups.** "When is it paid?" has no retrievable content on its own, so the previous question is folded into the retrieval query.
