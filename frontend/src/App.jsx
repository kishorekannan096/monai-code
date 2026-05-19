import { useState, useEffect } from 'react'
import './index.css'

const API_BASE = "http://localhost:8070";

function App() {
  const [sample, setSample] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [results, setResults] = useState(null);
  const [classList, setClassList] = useState([]);
  const [selectedClass, setSelectedClass] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    fetchClasses();
  }, []);

  const fetchClasses = async () => {
    try {
      const resp = await fetch(`${API_BASE}/classes`);
      if (resp.ok) {
        const data = await resp.json();
        setClassList(data.class_names);
        if (!selectedClass) setSelectedClass(data.class_names[0]);
      }
    } catch (err) {
      console.error("Failed to fetch classes:", err);
    }
  };

  const fetchSample = async () => {
    setLoading(true);
    setError("");
    setResults(null);
    setUploadedFile(null);
    setPreviewUrl("");
    try {
      const resp = await fetch(`${API_BASE}/sample`);
      if (!resp.ok) throw new Error("Failed to fetch sample");
      const data = await resp.json();
      setSample(data);
      // Set default selected class from first positive label if any, or just first label
      const firstPos = data.class_names.find((_, i) => data.labels[i] > 0.5);
      setSelectedClass(firstPos || data.class_names[0]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSample(null);
    setUploadedFile(file);
    setResults(null);
    setError("");

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const runInference = async () => {
    if ((!sample && !uploadedFile) || !selectedClass) return;
    setInferring(true);
    setError("");
    try {
      let resp;
      if (uploadedFile) {
        const formData = new FormData();
        formData.append("file", uploadedFile);
        formData.append("class_name", selectedClass);
        resp = await fetch(`${API_BASE}/infer_upload`, {
          method: "POST",
          body: formData
        });
      } else {
        resp = await fetch(`${API_BASE}/infer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_path: sample.image_path,
            class_name: selectedClass
          })
        });
      }

      if (!resp.ok) {
        const errData = await resp.json();
        throw new Error(errData.detail || "Inference failed");
      }
      const data = await resp.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setInferring(false);
    }
  };

  // Get display path for image
  const getFullUrl = (path) => {
    if (!path) return "";
    // If it's a static path from the backend
    if (path.startsWith("/static")) return `${API_BASE}${path}`;
    // If it's the raw local path from the sample, we need to map it?
    // User said they share a volume. If the backend serves data/ as well...
    // Let's assume the backend serves the whole project root or data folder.
    // In our main.py, we only mount /static. 
    // If image_path is like "data/images/...", we might need to mount that too.
    return `${API_BASE}/static/${path}`; // Assuming data is symlinked or mounted under static
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>MONAI Inference Server</h1>
        <p>Chest X-Ray Multi-label Classification with Grad-CAM</p>
      </header>

      <main className="main-container">
        {/* Left Col: Sample generation and selection */}
        <section className="card image-card">
          <div className="image-display">
            {loading ? (
              <div className="image-placeholder">
                <div className="loader"></div>
                <p>Loading random sample...</p>
              </div>
            ) : previewUrl ? (
              <img src={previewUrl} alt="Uploaded X-ray" />
            ) : sample ? (
              <img
                src={`${API_BASE}/images/${sample.image_path.split('/').pop()}`}
                alt="X-ray Sample"
              />
            ) : (
              <div className="image-placeholder">
                <p>No image loaded. Upload one or generate a sample.</p>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', width: '100%' }}>
            <button className="btn btn-primary" style={{ flex: 1 }} onClick={fetchSample} disabled={loading || inferring}>
              {loading ? "Loading..." : "Auto Fill (Sample)"}
            </button>
            <label className="btn btn-primary" style={{ flex: 1, textAlign: 'center', cursor: 'pointer' }}>
              Upload Image
              <input type="file" hidden accept="image/*" onChange={handleFileUpload} disabled={loading || inferring} />
            </label>
          </div>

          {(sample || uploadedFile) && (
            <div style={{ width: '100%' }}>
              {sample ? (
                <>
                  <h3>Ground Truth Labels</h3>
                  <div className="labels-grid">
                    {sample.class_names.map((name, i) => (
                      sample.labels[i] > 0.5 && (
                        <span key={name} className="label-badge active">{name}</span>
                      )
                    ))}
                    {!sample.labels.some(l => l > 0.5) && (
                      <span className="label-badge">No Finding</span>
                    )}
                  </div>
                </>
              ) : (
                <div className="info-badge" style={{ marginTop: '1rem' }}>
                  Ground truth not available for uploaded images.
                </div>
              )}

              <div style={{ marginTop: '1.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                  Target Class for Grad-CAM:
                </label>
                <select
                  className="class-selector"
                  value={selectedClass}
                  onChange={(e) => setSelectedClass(e.target.value)}
                >
                  {(classList.length > 0 ? classList : (sample ? sample.class_names : [])).map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </div>

              <button
                className="btn btn-primary"
                style={{ width: '100%', marginTop: '1rem', background: 'var(--success-color)' }}
                onClick={runInference}
                disabled={inferring}
              >
                {inferring ? <><div className="loader"></div> Running Inference...</> : "Submit for Inference"}
              </button>
            </div>
          )}
        </section>

        {/* Right Col: Results and Heatmap */}
        <section className="card results-card">
          {inferring && !results ? (
            <div className="image-placeholder">
              <div className="loader"></div>
              <p>Analyzing image and generating heatmaps...</p>
            </div>
          ) : results ? (
            <>
              <h3>Analysis Results</h3>
              <div className="prediction-banner">
                <strong>Model Prediction</strong>
                <p>{results.prediction}</p>
              </div>

              <div className="image-display" style={{ height: 'auto' }}>
                <img src={`${API_BASE}${results.heatmap_url}`} alt="Grad-CAM Heatmap" />
              </div>
              <p style={{ textAlign: 'center', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                Grad-CAM Heatmap for: <strong>{selectedClass}</strong>
              </p>

              <div>
                <h4>Top Probabilities</h4>
                <div className="probs-list">
                  {Object.entries(results.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([name, prob]) => (
                      <div key={name} className="prob-item">
                        <div className="prob-header">
                          <span>{name}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="prob-bar-bg">
                          <div
                            className="prob-bar-fill"
                            style={{ width: `${prob * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </>
          ) : (
            <div className="image-placeholder">
              {error ? (
                <div style={{ color: 'var(--danger-color)' }}>
                  <p>Error: {error}</p>
                </div>
              ) : (
                <p>Run inference to see results and Grad-CAM visualization.</p>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
