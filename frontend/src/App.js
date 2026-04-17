import React, { useMemo, useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const handleUpload = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setError(null);
    setData(null);
  };

  const handleAnalyze = async () => {
    if (!image) {
      setError("Please upload an image first.");
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append("image", image);

      const response = await fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Server error. Please try again.");
      }

      const json = await response.json();
      setData(json);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const graphData = useMemo(() => {
    if (!data || !data.people || data.people.length < 2) {
      return { nodes: [], edges: [] };
    }

    const centerX = 420;
    const centerY = 170;
    const radius = 120;

    const nodes = data.people.map((p, index) => {
      const angle = (2 * Math.PI * index) / data.people.length;
      return {
        id: String(p.person_id),
        label: `P${p.person_id}`,
        role: p.role,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
      };
    });

    const nodeMap = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const edges = (data.relationships || [])
      .map((r, idx) => {
        const source = nodeMap[String(r.p1)];
        const target = nodeMap[String(r.p2)];
        if (!source || !target) return null;
        return { id: `e-${idx}`, source, target, relation: r.relation };
      })
      .filter(Boolean);

    return { nodes, edges };
  }, [data]);

  return (
    <div className="app">
      <div className="container">
        <header className="hero">
          <p className="eyebrow">AI-Enabled Family Insight</p>
          <h1>Social Relationship Analysis</h1>
          <p className="subtext">
            Upload a group photo to detect people, infer social roles, and visualize relationship connections.
          </p>
        </header>

        <section className="panel">
          <div className="controls-row">
            <label className="upload-box">
              <input type="file" onChange={handleUpload} hidden />
              Upload Image
            </label>
            <button className="analyze-button" onClick={handleAnalyze} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>

          {error && <div className="error">{error}</div>}

          <div className="images-grid">
            {preview && (
              <div className="media-card">
                <h3>Uploaded Image</h3>
                <div className="image-section">
                  <img src={preview} alt="preview" />
                </div>
              </div>
            )}

            {data?.annotated_image && (
              <div className="media-card">
                <h3>Annotated Image (Face IDs)</h3>
                <div className="image-section">
                  <img src={data.annotated_image} alt="annotated faces" />
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="panel results">
          {!data && !error && <p className="placeholder">Results will appear here after analysis.</p>}

          {data && (
            <>
              <div className="results-grid">
                <div className="result-card">
                  <h3>Detected People</h3>
                  {data.people && data.people.length === 0 && <p>No faces detected.</p>}
                  {data.people && data.people.length > 0 && (
                    <ul>
                      {data.people.map((p) => (
                        <li key={p.person_id}>
                          <span className="pill">ID {p.person_id}</span>
                          Age: {p.age} | Gender: {p.gender} | Emotion: {p.emotion} | Role: {p.role}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <div className="result-card">
                  <h3>Relationships</h3>
                  {data.relationships && data.relationships.length === 0 && <p>No relationships inferred.</p>}
                  {data.relationships && data.relationships.length > 0 && (
                    <ul>
                      {data.relationships.map((r, idx) => (
                        <li key={idx}>
                          <span className="pill">P{r.p1} ↔ P{r.p2}</span>
                          {r.relation}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>

              {data.people && data.people.length > 1 && (
                <div className="result-card graph-card">
                  <h3>Relationship Graph (Undirected)</h3>
                  <div className="graph-box">
                    <svg viewBox="0 0 840 340" className="graph-svg">
                      {graphData.edges.map((edge) => (
                        <g key={edge.id}>
                          <line
                            x1={edge.source.x}
                            y1={edge.source.y}
                            x2={edge.target.x}
                            y2={edge.target.y}
                            stroke="#4b5563"
                            strokeWidth="2"
                          />
                          <text
                            x={(edge.source.x + edge.target.x) / 2}
                            y={(edge.source.y + edge.target.y) / 2 - 8}
                            textAnchor="middle"
                            fontSize="11"
                            fill="#111827"
                          >
                            {edge.relation}
                          </text>
                        </g>
                      ))}

                      {graphData.nodes.map((node) => (
                        <g key={node.id}>
                          <circle cx={node.x} cy={node.y} r="24" fill="#2563eb" />
                          <text x={node.x} y={node.y + 4} textAnchor="middle" fontSize="12" fill="#ffffff">
                            {node.label}
                          </text>
                          <text x={node.x} y={node.y + 40} textAnchor="middle" fontSize="11" fill="#374151">
                            {node.role}
                          </text>
                        </g>
                      ))}
                    </svg>
                  </div>
                </div>
              )}

              {data.summary && (
                <div className="summary-box">
                  <strong>Final Summary:</strong> {data.summary}
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;