
import { useState } from "react";
import {
  BrainCircuit,
  Send,
  Search,
  FileText,
  ChevronDown,
  ChevronUp,
  LoaderCircle,
  RotateCcw,
  Server,
  AlertCircle,
  CheckCircle2,
  Layers3,
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/api";

const strategies = [
  {
    value: "basic_vector",
    label: "Basic Vector",
    description: "Dense vector similarity retrieval",
  },
  {
    value: "semantic_vector",
    label: "Semantic Vector",
    description: "Semantic chunking with vector retrieval",
  },
  {
    value: "hybrid",
    label: "Hybrid Search",
    description: "BM25 keyword + vector retrieval using RRF",
  },
  {
    value: "hybrid_reranked",
    label: "Hybrid + Reranking",
    description: "Hybrid retrieval with cross-encoder reranking",
  },
];

function App() {
  const [question, setQuestion] = useState("");
  const [strategy, setStrategy] = useState("basic_vector");
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showContext, setShowContext] = useState(false);

  const selectedStrategy = strategies.find(
    (item) => item.value === strategy
  );

  async function checkResponse(response) {
    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || data.message || "The request could not be completed."
      );
    }

    return data;
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setShowContext(false);

    try {
      const response = await fetch(`${API_URL}/query/answer-strategy`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
          strategy,
          top_k: Number(topK),
        }),
      });

      const data = await checkResponse(response);
      setResult(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to the OptiRAG backend. Make sure FastAPI is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setQuestion("");
    setResult(null);
    setError("");
    setShowContext(false);
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <BrainCircuit size={23} />
          </div>
          <div>
            <h1>OptiRAG</h1>
            <span>Retrieval Intelligence Workspace</span>
          </div>
        </div>

        <div className="workspace-status">
          <span className="status-dot" />
          <span>Query Workspace</span>
        </div>
      </header>

      <main className="workspace">
        <section className="workspace-heading">
          <div className="eyebrow">
            <Layers3 size={15} />
            MICROSOFT AZURE DOCUMENTATION
          </div>

          <h2>Ask your documentation.</h2>
          <p>
            Explore Azure documentation using different retrieval
            strategies and evidence-grounded AI answers.
          </p>
        </section>

        <section className="query-card">
          <form onSubmit={handleSubmit}>
            <div className="field-heading">
              <label htmlFor="strategy">Retrieval strategy</label>
              <span>Choose how documents are retrieved</span>
            </div>

            <div className="strategy-grid">
              {strategies.map((item) => (
                <button
                  type="button"
                  key={item.value}
                  className={`strategy-option ${
                    strategy === item.value ? "selected" : ""
                  }`}
                  onClick={() => setStrategy(item.value)}
                >
                  <div className="strategy-radio">
                    {strategy === item.value && (
                      <span />
                    )}
                  </div>

                  <div className="strategy-info">
                    <strong>{item.label}</strong>
                    <small>{item.description}</small>
                  </div>
                </button>
              ))}
            </div>

            <div className="query-controls">
              <div className="field-heading">
                <label htmlFor="question">Your question</label>
                <span>Ask a question about Azure</span>
              </div>

              <div className="question-input">
                <textarea
                  id="question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="e.g. What is Azure Blob Storage and how does it work?"
                  rows={4}
                  disabled={loading}
                />

                <div className="input-footer">
                  <span>Answers are generated using retrieved evidence.</span>
                  <span>{question.length} characters</span>
                </div>
              </div>
            </div>

            <div className="form-footer">
              <div className="topk-control">
                <label htmlFor="topK">Retrieved chunks (Top-K)</label>
                <select
                  id="topK"
                  value={topK}
                  onChange={(event) => setTopK(event.target.value)}
                  disabled={loading}
                >
                  {[3, 5, 8, 10, 15, 20].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-actions">
                <button
                  type="button"
                  className="clear-button"
                  onClick={handleClear}
                  disabled={loading}
                >
                  <RotateCcw size={15} />
                  Clear
                </button>

                <button
                  type="submit"
                  className="submit-button"
                  disabled={loading || !question.trim()}
                >
                  {loading ? (
                    <>
                      <LoaderCircle className="spin" size={17} />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Send size={16} />
                      Run query
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>
        </section>

        {error && (
          <div className="error-message">
            <AlertCircle size={19} />
            <div>
              <strong>Query unsuccessful</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {loading && (
          <section className="loading-card">
            <LoaderCircle className="spin loading-icon" size={30} />
            <h3>Searching documentation</h3>
            <p>
              Retrieving relevant chunks and generating an
              evidence-grounded answer using {selectedStrategy.label}.
            </p>
          </section>
        )}

        {result && (
          <section className="results-section">
            <div className="results-heading">
              <div>
                <div className="eyebrow">
                  <CheckCircle2 size={15} />
                  QUERY COMPLETED
                </div>
                <h2>Answer</h2>
              </div>
              <span className="success-tag">Evidence generated</span>
            </div>

            <div className="answer-card">
              <div className="answer-card-header">
                <div className="result-icon">
                  <BrainCircuit size={19} />
                </div>
                <div>
                  <strong>Generated answer</strong>
                  <small>{result.model || "Language model"}</small>
                </div>
              </div>

              <div className="answer-content">
                {result.answer}
              </div>

              <div className="answer-metadata">
                <span>
                  Strategy:{" "}
                  <strong>
                    {strategies.find(
                      (item) =>
                        item.value === result.retrieval_strategy
                    )?.label || result.retrieval_strategy}
                  </strong>
                </span>
                <span>
                  Retrieved chunks: <strong>{result.result_count}</strong>
                </span>
              </div>
            </div>

            <div className="sources-section">
              <div className="section-title">
                <div>
                  <FileText size={18} />
                  <h3>Sources & citations</h3>
                </div>
                <span>
                  {result.citations?.length || 0} documents
                </span>
              </div>

              {result.citations?.length > 0 ? (
                <div className="citation-list">
                  {result.citations.map((citation) => (
                    <article
                      className="citation-card"
                      key={citation.document_id || citation.source_id}
                    >
                      <div className="citation-number">
                        {citation.source_id}
                      </div>

                      <div className="citation-details">
                        <strong>{citation.title}</strong>
                        <small>
                          {citation.relative_path || "Microsoft documentation"}
                        </small>
                        <div className="citation-meta">
                          <span>
                            Chunks:{" "}
                            {citation.chunk_indices?.length || 0}
                          </span>
                          <span>
                            {citation.source_url_status || "local_only"}
                          </span>
                        </div>

                        {citation.source_url && (
                          <a
                            href={citation.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Open source document
                          </a>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <div className="empty-sources">
                  No document citations were returned for this answer.
                </div>
              )}
            </div>

            <div className="context-section">
              <button
                className="context-toggle"
                onClick={() => setShowContext(!showContext)}
              >
                <div>
                  <Search size={17} />
                  <span>Retrieved context</span>
                  <small>
                    {result.retrieved_context?.length || 0} chunks
                  </small>
                </div>

                {showContext ? (
                  <ChevronUp size={18} />
                ) : (
                  <ChevronDown size={18} />
                )}
              </button>

              {showContext && (
                <div className="context-list">
                  {result.retrieved_context?.length > 0 ? (
                    result.retrieved_context.map((chunk, index) => (
                      <article className="context-card" key={index}>
                        <div className="context-card-header">
                          <strong>
                            {chunk.metadata?.title || `Retrieved chunk ${index + 1}`}
                          </strong>
                          <span>
                            Score:{" "}
                            {typeof chunk.score === "number"
                              ? chunk.score.toFixed(4)
                              : "N/A"}
                          </span>
                        </div>
                        <p>{chunk.text}</p>
                      </article>
                    ))
                  ) : (
                    <p className="empty-sources">
                      No retrieved context was returned.
                    </p>
                  )}
                </div>
              )}
            </div>
          </section>
        )}

        <footer className="workspace-footer">
          <div>
            <Server size={14} />
            <span>OptiRAG API · {API_URL}</span>
          </div>
          <span>Production-oriented RAG system</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
