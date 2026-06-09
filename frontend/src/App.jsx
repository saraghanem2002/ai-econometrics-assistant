import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [preview, setPreview] = useState([]);
  const [objective, setObjective] = useState("");
  const [dependent, setDependent] = useState("");
  const [independent, setIndependent] = useState([]);
  const [fullAnalysis, setFullAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const API_URL = "http://127.0.0.1:8000";

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post(`${API_URL}/upload`, formData);
    setColumns(response.data.columns);
    setPreview(response.data.preview);
    setFullAnalysis(null);
  };

  const runFullAnalysis = async () => {
    if (!file || !objective) {
      alert("Please upload a file, write an objective, and select variables.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("objective", objective);
    formData.append("dependent_variable", dependent);
    formData.append("independent_variables", independent.join(","));

    const formData2 = new FormData();
    formData2.append("file", file);
    formData2.append("objective", objective);

    const response = await axios.post(`${API_URL}/execute-ai-plan`, formData2);

    const data = response.data;
    console.log("FULL RESPONSE:", JSON.stringify(response.data));
    console.log("RAW ai_plan:", data.ai_plan);
    console.log("TYPE:", typeof data.ai_plan);
    if (typeof data.ai_plan === "string") {
      data.ai_plan = JSON.parse(data.ai_plan);
    }
    
    setFullAnalysis(data);
    setLoading(false);
  };

  const toggleIndependent = (col) => {
    if (independent.includes(col)) {
      setIndependent(independent.filter((item) => item !== col));
    } else {
      setIndependent([...independent, col]);
    }
  };

  return (
    <div className="container">
      <h1>AI Econometrics Assistant</h1>

      <p className="subtitle">
        Upload a dataset, define your research objective, and generate a full AI econometric report.
      </p>

      <div className="card">
        <h2>1. Upload Dataset</h2>

        <input
          type="file"
          accept=".csv,.xlsx"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button onClick={handleUpload}>Upload & Preview</button>
      </div>

      {preview.length > 0 && (
        <div className="card">
          <h2>2. Data Preview</h2>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  {Object.keys(preview[0]).map((key) => (
                    <th key={key}>{key}</th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {preview.map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value, i) => (
                      <td key={i}>{value}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {columns.length > 0 && (
        <div className="card">
          <h2>3. Research Objective</h2>

          <textarea
            placeholder="Example: Analyze the effect of inflation and interest rates on investment in Egypt"
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
          />

          <h2>4. Select Variables</h2>

          <label>Dependent Variable (Y)</label>
          <select value={dependent} onChange={(e) => setDependent(e.target.value)}>
            <option value="">Select dependent variable</option>
            {columns.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>

          <label>Independent Variables (X)</label>
          <div className="checkbox-list">
            {columns
              .filter((col) => col !== dependent)
              .map((col) => (
                <label key={col} className="checkbox-item">
                  <input
                    type="checkbox"
                    checked={independent.includes(col)}
                    onChange={() => toggleIndependent(col)}
                  />
                  {col}
                </label>
              ))}
          </div>

          <button onClick={runFullAnalysis}>
            {loading ? "Running Full AI Analysis..." : "Run Full AI Analysis"}
          </button>
        </div>
      )}

      {fullAnalysis && (
  <div className="card results">
    <h2>5. AI Econometric Plan</h2>

    <p><strong>Dataset Type:</strong> {fullAnalysis.ai_plan.dataset_type}</p>
    <p><strong>Dependent Variable:</strong> {fullAnalysis.ai_plan.dependent_variable}</p>
    <p>
      <strong>Independent Variables:</strong>{" "}
      {fullAnalysis.ai_plan.independent_variables.join(", ")}
    </p>

    <p><strong>Reasoning:</strong> {fullAnalysis.ai_plan.reasoning}</p>

    <h2>6. Executed Results</h2>

    <p>
      <strong>R²:</strong>{" "}
      {fullAnalysis.executed_results.models.ols.r_squared.toFixed(3)}
    </p>

    <p>
      <strong>Adjusted R²:</strong>{" "}
      {fullAnalysis.executed_results.models.ols.adjusted_r_squared.toFixed(3)}
    </p>

    <h3>OLS Coefficients</h3>

    <table>
      <thead>
        <tr>
          <th>Variable</th>
          <th>Coefficient</th>
          <th>P-value</th>
        </tr>
      </thead>

      <tbody>
        {Object.keys(fullAnalysis.executed_results.models.ols.coefficients).map((key) => (
          <tr key={key}>
            <td>{key}</td>
            <td>{fullAnalysis.executed_results.models.ols.coefficients[key].toFixed(4)}</td>
            <td>{fullAnalysis.executed_results.models.ols.p_values[key].toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>

    <h3>Executed Tests</h3>

    <p>
      <strong>Tests selected by AI:</strong>{" "}
      {fullAnalysis.ai_plan.recommended_tests.join(", ")}
    </p>

    <h3>VIF</h3>

    <table>
      <thead>
        <tr>
          <th>Variable</th>
          <th>VIF</th>
        </tr>
      </thead>

      <tbody>
        {fullAnalysis.executed_results.tests.vif.map((item) => (
          <tr key={item.variable}>
            <td>{item.variable}</td>
            <td>{item.vif.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>

    <h3>ADF Test</h3>

    <table>
      <thead>
        <tr>
          <th>Variable</th>
          <th>P-value</th>
          <th>Stationary?</th>
        </tr>
      </thead>

      <tbody>
        {Object.keys(fullAnalysis.executed_results.tests.adf).map((key) => (
          <tr key={key}>
            <td>{key}</td>
            <td>{fullAnalysis.executed_results.tests.adf[key].p_value?.toFixed(4)}</td>
            <td>
              {fullAnalysis.executed_results.tests.adf[key].is_stationary_5_percent
                ? "Yes"
                : "No"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>

    <p>
      <strong>Durbin-Watson:</strong>{" "}
      {fullAnalysis.executed_results.tests.durbin_watson.statistic.toFixed(3)}
    </p>

    <p>
      <strong>Breusch-Pagan p-value:</strong>{" "}
      {fullAnalysis.executed_results.tests.breusch_pagan.lm_p_value.toFixed(4)}
    </p>

    <h2>7. Final AI Report</h2>
    <pre className="report">{fullAnalysis.final_report}</pre>
  </div>
)}
    </div>
  );
}

export default App;