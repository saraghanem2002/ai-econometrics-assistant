from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import statsmodels.api as sm
import io
import os

from dotenv import load_dotenv
import google.generativeai as genai

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import json


app = FastAPI(title="AI Econometrics Assistant")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_uploaded_file(file: UploadFile, content: bytes):
    if file.filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif file.filename.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(content))
    else:
        return None


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    df = read_uploaded_file(file, content)

    if df is None:
        return {"error": "Only CSV and Excel files are supported"}

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(10).to_dict(orient="records")
    }


@app.post("/regression")
async def run_regression(
    file: UploadFile = File(...),
    dependent_variable: str = Form(...),
    independent_variables: str = Form(...)
):
    content = await file.read()
    df = read_uploaded_file(file, content)

    if df is None:
        return {"error": "Only CSV and Excel files are supported"}

    x_vars = [x.strip() for x in independent_variables.split(",")]

    df = df[[dependent_variable] + x_vars].dropna()

    y = df[dependent_variable]
    X = df[x_vars]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return {
        "dependent_variable": dependent_variable,
        "independent_variables": x_vars,
        "observations": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "coefficients": model.params.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "summary": model.summary().as_text()
    }


@app.post("/analyze-dataset")
async def analyze_dataset(
    file: UploadFile = File(...),
    objective: str = Form("")
):
    content = await file.read()
    df = read_uploaded_file(file, content)

    if df is None:
        return {"error": "Only CSV and Excel files are supported"}

    rows, cols = df.shape

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()

    year_like_columns = []
    date_like_columns = []

    for col in df.columns:
        col_lower = col.lower()

        if "year" in col_lower:
            year_like_columns.append(col)

        if "date" in col_lower or "time" in col_lower:
            date_like_columns.append(col)

    missing_values = df.isnull().sum().to_dict()

    if len(year_like_columns) > 0 or len(date_like_columns) > 0:
        dataset_type = "Possible Time Series"
    else:
        dataset_type = "Possible Cross-Sectional Data"

    return {
        "objective": objective,
        "rows": rows,
        "columns_count": cols,
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_columns,
        "text_columns": text_columns,
        "year_like_columns": year_like_columns,
        "date_like_columns": date_like_columns,
        "missing_values": missing_values,
        "detected_dataset_type": dataset_type
    }


@app.post("/ai-econometric-plan")
async def ai_econometric_plan(
    file: UploadFile = File(...),
    objective: str = Form(...)
):
    content = await file.read()
    df = read_uploaded_file(file, content)

    if df is None:
        return {"error": "Only CSV and Excel files are supported"}

    metadata = {
        "rows": len(df),
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "text_columns": df.select_dtypes(include=["object"]).columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "basic_statistics": df.describe().to_dict()
    }

    prompt = f"""
You are an expert AI econometrics assistant.

User objective:
{objective}

Dataset metadata:
{metadata}

Give a professional econometric analysis plan:
1. Identify dataset type.
2. Explain why.
3. Suggest dependent and independent variables.
4. Recommend suitable models/tests.
5. Explain whether OLS is enough or only a baseline.
6. Mention econometric risks.
7. Do not invent numerical regression results.
"""

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)

        return {
            "objective": objective,
            "ai_plan": response.text
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/diagnostics")
async def run_diagnostics(
    file: UploadFile = File(...),
    dependent_variable: str = Form(...),
    independent_variables: str = Form(...)
):
    content = await file.read()
    df = read_uploaded_file(file, content)

    if df is None:
        return {"error": "Only CSV and Excel files are supported"}

    x_vars = [x.strip() for x in independent_variables.split(",")]

    selected_columns = [dependent_variable] + x_vars
    df = df[selected_columns].dropna()

    numeric_df = df.select_dtypes(include=["number"])

    results = {}

    results["correlation_matrix"] = numeric_df.corr().round(4).to_dict()

    vif_results = []

    if len(x_vars) >= 2:
        X_vif = df[x_vars].select_dtypes(include=["number"]).dropna()

        for i, col in enumerate(X_vif.columns):
            vif_results.append({
                "variable": col,
                "vif": float(variance_inflation_factor(X_vif.values, i))
            })

    results["vif"] = vif_results

    adf_results = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()

        if len(series) >= 8:
            adf_test = adfuller(series)

            adf_results[col] = {
                "adf_statistic": float(adf_test[0]),
                "p_value": float(adf_test[1]),
                "is_stationary_5_percent": bool(adf_test[1] < 0.05)
            }
        else:
            adf_results[col] = {
                "error": "Not enough observations for ADF test"
            }

    results["adf_test"] = adf_results

    y = df[dependent_variable]
    X = df[x_vars]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    residuals = model.resid

    results["durbin_watson"] = {
        "statistic": float(durbin_watson(residuals)),
        "interpretation": "Around 2 suggests no strong autocorrelation. Below 2 suggests positive autocorrelation. Above 2 suggests negative autocorrelation."
    }

    bp_test = het_breuschpagan(residuals, model.model.exog)

    results["breusch_pagan"] = {
        "lm_statistic": float(bp_test[0]),
        "lm_p_value": float(bp_test[1]),
        "f_statistic": float(bp_test[2]),
        "f_p_value": float(bp_test[3]),
        "heteroskedasticity_detected_5_percent": bool(bp_test[1] < 0.05)
    }

    return {
        "dependent_variable": dependent_variable,
        "independent_variables": x_vars,
        "observations_used": len(df),
        "diagnostics": results
    }


@app.post("/interpret-diagnostics")
async def interpret_diagnostics(diagnostics_results: dict):
    prompt = f"""
You are an expert econometrics assistant.

Interpret the following econometric diagnostics clearly and professionally.

Diagnostics results:
{diagnostics_results}

Explain:
1. Correlation results
2. VIF / multicollinearity
3. ADF stationarity results
4. Durbin-Watson autocorrelation result
5. Breusch-Pagan heteroskedasticity result
6. Whether OLS results are reliable
7. What the user should do next

Do not invent results. Only interpret the provided values.
"""

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)

        return {
            "ai_interpretation": response.text
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/full-analysis")
async def full_analysis(
    file: UploadFile = File(...),
    objective: str = Form(...),
    dependent_variable: str = Form(...),
    independent_variables: str = Form(...)
):
    content = await file.read()
    df = read_uploaded_file(file, content)

    if df is None:
        return {"error": "Only CSV and Excel files are supported"}

    x_vars = [x.strip() for x in independent_variables.split(",")]

    selected_columns = [dependent_variable] + x_vars
    df = df[selected_columns].dropna()

    numeric_df = df.select_dtypes(include=["number"])

    y = df[dependent_variable]
    X = df[x_vars]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    residuals = model.resid

    regression_results = {
        "dependent_variable": dependent_variable,
        "independent_variables": x_vars,
        "observations": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "coefficients": model.params.to_dict(),
        "p_values": model.pvalues.to_dict()
    }

    diagnostics = {}

    diagnostics["correlation_matrix"] = numeric_df.corr().round(4).to_dict()

    vif_results = []

    if len(x_vars) >= 2:
        X_vif = df[x_vars].select_dtypes(include=["number"]).dropna()

        for i, col in enumerate(X_vif.columns):
            vif_results.append({
                "variable": col,
                "vif": float(variance_inflation_factor(X_vif.values, i))
            })

    diagnostics["vif"] = vif_results

    adf_results = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()

        if len(series) >= 8:
            adf_test = adfuller(series)

            adf_results[col] = {
                "adf_statistic": float(adf_test[0]),
                "p_value": float(adf_test[1]),
                "is_stationary_5_percent": bool(adf_test[1] < 0.05)
            }
        else:
            adf_results[col] = {
                "error": "Not enough observations for ADF test"
            }

    diagnostics["adf_test"] = adf_results

    diagnostics["durbin_watson"] = {
        "statistic": float(durbin_watson(residuals))
    }

    bp_test = het_breuschpagan(residuals, model.model.exog)

    diagnostics["breusch_pagan"] = {
        "lm_statistic": float(bp_test[0]),
        "lm_p_value": float(bp_test[1]),
        "f_statistic": float(bp_test[2]),
        "f_p_value": float(bp_test[3]),
        "heteroskedasticity_detected_5_percent": bool(bp_test[1] < 0.05)
    }

    prompt = f"""
You are an expert AI econometrics assistant.

User objective:
{objective}

Regression results:
{regression_results}

Diagnostics:
{diagnostics}

Write a full professional econometric report.

Include:
1. Executive summary
2. Model specification
3. Interpretation of coefficients
4. Model fit
5. Correlation and multicollinearity interpretation
6. Stationarity interpretation
7. Autocorrelation interpretation
8. Heteroskedasticity interpretation
9. Whether OLS is reliable
10. Recommended next steps

Do not invent numbers. Only use the provided results.
"""

    try:
        ai_model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = ai_model.generate_content(prompt)
        ai_report = response.text

    except Exception as e:
        ai_report = f"AI report could not be generated. Error: {str(e)}"

    return {
        "objective": objective,
        "regression_results": regression_results,
        "diagnostics": diagnostics,
        "ai_report": ai_report
    }


@app.post("/ai-structured-plan")
async def ai_structured_plan(
    file: UploadFile = File(...),
    objective: str = Form(...)
):
    content = await file.read()
    df = read_uploaded_file(file, content)

    if df is None:
        return {"error": "Only CSV and Excel files are supported"}

    metadata = {
        "rows": len(df),
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "text_columns": df.select_dtypes(include=["object"]).columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "basic_statistics": df.describe().to_dict()
    }

    prompt = f"""
You are an expert econometrics assistant.

User objective:
{objective}

Dataset metadata:
{metadata}

Create an executable econometric analysis plan.

Return ONLY valid JSON. No markdown. No explanation outside JSON.

JSON format:
{{
  "dataset_type": "time_series OR cross_sectional OR panel_data",
  "dependent_variable": "column name",
  "independent_variables": ["column1", "column2"],
  "recommended_tests": ["correlation_matrix", "vif", "adf", "durbin_watson", "breusch_pagan"],
  "recommended_models": ["ols"],
  "reasoning": "short explanation of why these tests/models are recommended"
}}

Rules:
- Only use column names that exist in the dataset.
- Do not invent variables.
- If the objective clearly mentions variables, use them.
- If there is a Year/Date column, consider time_series.
- If there is an entity column plus Year/Date, consider panel_data.
- For now, only recommend models/tests from the allowed lists.
"""

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)

        return {
            "objective": objective,
            "metadata": metadata,
            "structured_plan": response.text
        }

    except Exception as e:
        return {"error": str(e)}
    

@app.post("/execute-ai-plan")
async def execute_ai_plan(
    file: UploadFile = File(...),
    objective: str = Form(...)
):
    content = await file.read()
    df = read_uploaded_file(file, content)

    if df is None:
        return {"error": "Only CSV and Excel files are supported"}

    metadata = {
        "rows": len(df),
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
        "text_columns": df.select_dtypes(include=["object"]).columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "basic_statistics": df.describe().to_dict()
    }

    planning_prompt = f"""
You are an expert econometrics assistant.

User objective:
{objective}

Dataset metadata:
{metadata}

Create an executable econometric analysis plan.

Return ONLY valid JSON. No markdown.

JSON format:
{{
  "dataset_type": "time_series OR cross_sectional OR panel_data",
  "dependent_variable": "column name",
  "independent_variables": ["column1", "column2"],
  "recommended_tests": ["correlation_matrix", "vif", "adf", "durbin_watson", "breusch_pagan"],
  "recommended_models": ["ols"],
  "reasoning": "short explanation"
}}

Rules:
- Only use column names that exist in the dataset.
- Do not invent variables.
- If the objective mentions variables, use them.
- For now, only recommend tests/models from the allowed lists.
"""

    try:
        ai_model = genai.GenerativeModel("models/gemini-2.5-flash")
        plan_response = ai_model.generate_content(planning_prompt)
        plan_text = plan_response.candidates[0].content.parts[0].text.strip()

        import re
        plan_text = re.sub(r"```json|```", "", plan_text).strip()

        plan = json.loads(plan_text)

    except Exception as e:
        return {
            "error": "AI planning failed",
            "details": str(e)
        }

    dependent_variable = plan["dependent_variable"]
    x_vars = plan["independent_variables"]

    selected_columns = [dependent_variable] + x_vars
    df_model = df[selected_columns].dropna()
    numeric_df = df_model.select_dtypes(include=["number"])

    executed_results = {
        "plan": plan,
        "models": {},
        "tests": {}
    }

    residuals = None
    model = None

    if "ols" in plan["recommended_models"]:
        y = df_model[dependent_variable]
        X = df_model[x_vars]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        residuals = model.resid

        executed_results["models"]["ols"] = {
            "dependent_variable": dependent_variable,
            "independent_variables": x_vars,
            "observations": int(model.nobs),
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "coefficients": model.params.to_dict(),
            "p_values": model.pvalues.to_dict()
        }

    if "correlation_matrix" in plan["recommended_tests"]:
        executed_results["tests"]["correlation_matrix"] = (
            numeric_df.corr().round(4).to_dict()
        )

    if "vif" in plan["recommended_tests"]:
        vif_results = []

        if len(x_vars) >= 2:
            X_vif = df_model[x_vars].select_dtypes(include=["number"]).dropna()

            for i, col in enumerate(X_vif.columns):
                vif_results.append({
                    "variable": col,
                    "vif": float(variance_inflation_factor(X_vif.values, i))
                })

        executed_results["tests"]["vif"] = vif_results

    if "adf" in plan["recommended_tests"]:
        adf_results = {}

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()

            if len(series) >= 8:
                adf_test = adfuller(series)

                adf_results[col] = {
                    "adf_statistic": float(adf_test[0]),
                    "p_value": float(adf_test[1]),
                    "is_stationary_5_percent": bool(adf_test[1] < 0.05)
                }
            else:
                adf_results[col] = {
                    "error": "Not enough observations for ADF test"
                }

        executed_results["tests"]["adf"] = adf_results

    if "durbin_watson" in plan["recommended_tests"] and residuals is not None:
        executed_results["tests"]["durbin_watson"] = {
            "statistic": float(durbin_watson(residuals))
        }

    if "breusch_pagan" in plan["recommended_tests"] and residuals is not None and model is not None:
        bp_test = het_breuschpagan(residuals, model.model.exog)

        executed_results["tests"]["breusch_pagan"] = {
            "lm_statistic": float(bp_test[0]),
            "lm_p_value": float(bp_test[1]),
            "f_statistic": float(bp_test[2]),
            "f_p_value": float(bp_test[3]),
            "heteroskedasticity_detected_5_percent": bool(bp_test[1] < 0.05)
        }

    interpretation_prompt = f"""
You are an expert econometrics assistant.

User objective:
{objective}

The AI first created this econometric plan:
{plan}

Python executed the selected models/tests and produced:
{executed_results}

Write a professional final econometric report.

Explain:
1. Why this plan was selected
2. What models/tests were executed
3. Main statistical findings
4. Whether OLS is reliable
5. Econometric problems detected
6. Recommended next steps

Do not invent results. Only interpret the executed outputs.
"""

    try:
        final_response = ai_model.generate_content(interpretation_prompt)
        final_report = final_response.candidates[0].content.parts[0].text

    except Exception as e:
        final_report = f"AI final interpretation failed: {str(e)}"

    return {
        "objective": objective,
        "ai_plan": plan,
        "executed_results": executed_results,
        "final_report": final_report
    }