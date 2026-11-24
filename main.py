from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import uuid
import os
from io import BytesIO

from data_processor import load_csv, profile_data
from clustering import run_clustering
from ai_explainer import (
    analyze_clustering,
    suggest_feature_adjustments
)

from dotenv import load_dotenv

load_dotenv()
# --------------------------------------------------
# APP INITIALIZATION
# --------------------------------------------------

app = FastAPI(title="HCP Segmentation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Temporary in-memory dataset storage
datasets = {}


# --------------------------------------------------
# BEDROCK CONFIGURATION CHECK
# --------------------------------------------------

def check_bedrock_configuration():

    if not os.getenv("BEDROCK_MODEL_ID"):

        raise HTTPException(
            status_code=500,
            detail=(
                "Bedrock model is not configured. "
                "Set BEDROCK_MODEL_ID before starting the backend."
            )
        )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "HCP Segmentation API running",
        "status": "active"
    }


# --------------------------------------------------
# 1. UPLOAD DATASET
# --------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Upload a CSV file."
        )

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    try:

        df = load_csv(BytesIO(content))

        if df.empty:

            raise ValueError(
                "The CSV contains no records."
            )

        profile = profile_data(df)

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not process CSV: {str(e)}"
        )

    dataset_id = str(uuid.uuid4())

    datasets[dataset_id] = {
        "df": df,
        "profile": profile,
        "filename": file.filename
    }

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "filename": file.filename,
        "profile": profile
    }


# --------------------------------------------------
# 2. CLUSTERING REQUEST
# --------------------------------------------------

class SegmentRequest(BaseModel):

    dataset_id: str

    selected_features: List[str]

    weights: Dict[str, float] = Field(
        default_factory=dict
    )

    n_clusters: int = Field(
        default=3,
        ge=2,
        le=20
    )


@app.post("/segment")
def segment(request: SegmentRequest):

    item = datasets.get(request.dataset_id)

    if not item:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    if not request.selected_features:

        raise HTTPException(
            status_code=400,
            detail="Select at least one feature."
        )

    # Only allow features approved by the data profiling stage
    allowed_features = {
        feature["column"]
        for feature in item["profile"]["features"]
    }

    invalid_features = [
        feature
        for feature in request.selected_features
        if feature not in allowed_features
    ]

    if invalid_features:

        raise HTTPException(
            status_code=400,
            detail=(
                "The following columns cannot be used for clustering: "
                f"{invalid_features}"
            )
        )

    # Validate feature weights
    for feature, weight in request.weights.items():

        if feature not in request.selected_features:

            raise HTTPException(
                status_code=400,
                detail=f"Weight supplied for unselected feature: {feature}"
            )

        if not 0 <= weight <= 5:

            raise HTTPException(
                status_code=400,
                detail=f"Weight for {feature} must be between 0 and 5."
            )

    try:

        result = run_clustering(
            item["df"],
            request.selected_features,
            request.weights,
            request.n_clusters
        )

        item["result"] = result

        item["selected_features"] = request.selected_features

        item["weights"] = request.weights

        item["n_clusters"] = request.n_clusters

        # Remove previous AI results and proposals
        item.pop("analysis", None)
        item.pop("adjustment_suggestion", None)

        return {
            "status": "success",
            **result
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Clustering failed: {str(e)}"
        )


# --------------------------------------------------
# 3. AI CLUSTERING ANALYSIS - AMAZON BEDROCK
# --------------------------------------------------

@app.get("/analyze/{dataset_id}")
def analyze_results(dataset_id: str):

    item = datasets.get(dataset_id)

    if not item:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    if "result" not in item:

        raise HTTPException(
            status_code=400,
            detail="Run clustering before requesting AI analysis."
        )

    check_bedrock_configuration()

    try:

        # Send actual clustering results to Amazon Bedrock
        explanation = analyze_clustering(
            item["result"]
        )

        item["analysis"] = explanation

        return {
            "status": "success",
            "analysis": explanation
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {str(e)}"
        )


# --------------------------------------------------
# 4. AI NATURAL LANGUAGE FEATURE ADJUSTMENTS
# --------------------------------------------------

class AdjustmentRequest(BaseModel):

    dataset_id: str

    instruction: str = Field(
        min_length=3,
        max_length=1000
    )

    selected_features: List[str]

    current_weights: Dict[str, float] = Field(
        default_factory=dict
    )


@app.post("/suggest-adjustments")
def suggest_adjustments(request: AdjustmentRequest):

    item = datasets.get(request.dataset_id)

    if not item:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    if "result" not in item:

        raise HTTPException(
            status_code=400,
            detail="Run clustering before requesting adjustments."
        )

    check_bedrock_configuration()

    # Only allow adjustment of features used in the
    # latest clustering run.
    used_features = item.get(
        "selected_features",
        []
    )

    if set(request.selected_features) != set(used_features):

        raise HTTPException(
            status_code=400,
            detail=(
                "Selected features do not match "
                "the latest clustering run."
            )
        )

    try:

        suggestion = suggest_feature_adjustments(
            instruction=request.instruction,
            selected_features=used_features,
            current_weights=request.current_weights
        )

        # Store proposal for reference.
        # It is not applied until the user approves it
        # and sends a new /segment request.
        item["adjustment_suggestion"] = suggestion

        return {
            "status": "success",
            "proposed_weights": suggestion["proposed_weights"],
            "explanation": suggestion["explanation"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not generate adjustment: {str(e)}"
        )


# --------------------------------------------------
# 5. HELPER: CREATE EXCEL CLUSTER PROFILE DATA
# --------------------------------------------------

def build_profile_dataframe(methods):

    rows = []

    for method in methods:

        profiles = method.get(
            "cluster_profiles",
            {}
        )

        for cluster_name, cluster_data in profiles.items():

            row = {
                "Algorithm": method["method"],
                "Cluster": cluster_name,
                "HCP Count": cluster_data.get("size", 0)
            }

            feature_profiles = cluster_data.get(
                "profile",
                {}
            )

            for feature, values in feature_profiles.items():

                if "mean" in values:

                    row[f"{feature}_Mean"] = values.get("mean")
                    row[f"{feature}_Median"] = values.get("median")
                    row[f"{feature}_Std"] = values.get("std")

                elif "distribution" in values:

                    row[f"{feature}_Distribution"] = str(
                        values["distribution"]
                    )

            rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# 6. HELPER: CREATE HCP ASSIGNMENT DATA
# --------------------------------------------------

def build_assignment_dataframe(methods):

    frames = []

    for method in methods:

        assignments = method.get(
            "hcp_assignments",
            []
        )

        if not assignments:
            continue

        df_assignments = pd.DataFrame(assignments)

        df_assignments.insert(
            0,
            "Algorithm",
            method["method"]
        )

        frames.append(df_assignments)

    if frames:

        return pd.concat(
            frames,
            ignore_index=True
        )

    return pd.DataFrame()


# --------------------------------------------------
# 7. DOWNLOAD EXCEL REPORT
# --------------------------------------------------

@app.get("/report/{dataset_id}")
def download_report(dataset_id: str):

    item = datasets.get(dataset_id)

    if not item or "result" not in item:

        raise HTTPException(
            status_code=404,
            detail="Run segmentation first."
        )

    output = BytesIO()

    result = item["result"]

    methods = result.get("methods", [])

    try:

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            # --------------------------------------
            # DATA SCHEMA
            # --------------------------------------

            pd.DataFrame(
                item["profile"]["columns"]
            ).to_excel(
                writer,
                sheet_name="Data Schema",
                index=False
            )

            # --------------------------------------
            # MODEL COMPARISON
            # --------------------------------------

            comparison_rows = []

            for method in methods:

                comparison_rows.append({

                    "Algorithm": method["method"],

                    "Silhouette Score": method.get(
                        "silhouette"
                    ),

                    "Clusters": method.get(
                        "clusters"
                    ),

                    "Noise Points": method.get(
                        "noise_points", 0
                    ),

                    "Cluster Sizes": str(
                        method.get("cluster_sizes", {})
                    ),

                    "Error": method.get("error", "")

                })

            pd.DataFrame(
                comparison_rows
            ).to_excel(
                writer,
                sheet_name="Model Comparison",
                index=False
            )

            # --------------------------------------
            # ELBOW ANALYSIS
            # --------------------------------------

            elbow = result.get(
                "elbow_analysis",
                {}
            )

            elbow_rows = elbow.get(
                "tested_values",
                []
            )

            if elbow_rows:

                pd.DataFrame(
                    elbow_rows
                ).to_excel(
                    writer,
                    sheet_name="Elbow Analysis",
                    index=False
                )

            # --------------------------------------
            # CLUSTER PROFILES
            # --------------------------------------

            profile_df = build_profile_dataframe(
                methods
            )

            if not profile_df.empty:

                profile_df.to_excel(
                    writer,
                    sheet_name="Cluster Profiles",
                    index=False
                )

            # --------------------------------------
            # HCP ASSIGNMENTS
            # --------------------------------------

            assignment_df = build_assignment_dataframe(
                methods
            )

            if not assignment_df.empty:

                assignment_df.to_excel(
                    writer,
                    sheet_name="HCP Assignments",
                    index=False
                )

            # --------------------------------------
            # K-MEANS / HIERARCHICAL COMPARISON
            # --------------------------------------

            agreement = result.get(
                "assignment_comparison",
                {}
            )

            if agreement:

                pd.DataFrame([
                    {
                        "Comparison": key,
                        "Result": value
                    }
                    for key, value in agreement.items()
                ]).to_excel(
                    writer,
                    sheet_name="Model Agreement",
                    index=False
                )

            # --------------------------------------
            # AI BUSINESS ANALYSIS
            # --------------------------------------

            if "analysis" in item:

                pd.DataFrame({
                    "AI Business Analysis": [
                        item["analysis"]
                    ]
                }).to_excel(
                    writer,
                    sheet_name="AI Analysis",
                    index=False
                )

            # --------------------------------------
            # APPROVED / PROPOSED WEIGHTS
            # --------------------------------------

            if "adjustment_suggestion" in item:

                suggestion = item[
                    "adjustment_suggestion"
                ]

                pd.DataFrame([
                    {
                        "Feature": feature,
                        "Proposed Weight": weight
                    }
                    for feature, weight
                    in suggestion.get(
                        "proposed_weights",
                        {}
                    ).items()
                ]).to_excel(
                    writer,
                    sheet_name="Weight Suggestions",
                    index=False
                )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not generate Excel report: {str(e)}"
        )

    output.seek(0)

    return StreamingResponse(
        output,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition":
            'attachment; filename="hcp_segmentation_report.xlsx"'
        }
    )
