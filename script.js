
const API = "http://127.0.0.1:8000";

let datasetId = null;
let selectedFeatures = [];
let currentWeights = {};
let proposedWeights = {};


// --------------------------------------------------
// HELPER FUNCTIONS
// --------------------------------------------------

function escapeHTML(value) {

    return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[char]);

}


async function apiRequest(url, options = {}) {

    const response = await fetch(url, options);

    let data;

    try {
        data = await response.json();
    } catch {
        data = {};
    }

    if (!response.ok) {

        throw new Error(
            data.detail || data.message || "Something went wrong."
        );

    }

    return data;

}


function setStatus(elementId, message, isError = false) {

    const element = document.getElementById(elementId);

    if (!element) return;

    element.textContent = message;

    element.style.color = isError ? "#dc2626" : "#475569";

}


// --------------------------------------------------
// 1. UPLOAD DATASET
// --------------------------------------------------

async function uploadDataset() {

    const fileInput = document.getElementById("fileInput");

    const file = fileInput.files[0];

    if (!file) {

        setStatus(
            "uploadStatus",
            "Please select a CSV file first.",
            true
        );

        return;

    }

    if (!file.name.toLowerCase().endsWith(".csv")) {

        setStatus(
            "uploadStatus",
            "Only CSV files are supported.",
            true
        );

        return;

    }

    const formData = new FormData();

    formData.append("file", file);

    try {

        setStatus(
            "uploadStatus",
            "Uploading and analysing dataset..."
        );

        const data = await apiRequest(
            `${API}/upload`,
            {
                method: "POST",
                body: formData
            }
        );

        datasetId = data.dataset_id;

        selectedFeatures = [];

        currentWeights = {};

        proposedWeights = {};

        document.getElementById("resultsSection").style.display = "none";

        showProfile(data.profile);

        setStatus(
            "uploadStatus",
            `Successfully uploaded ${data.filename}.`
        );

        setStatus("profileStatus", "");

        window.scrollTo({
            top: document.getElementById("profileSection").offsetTop - 30,
            behavior: "smooth"
        });

    } catch (error) {

        setStatus(
            "uploadStatus",
            error.message,
            true
        );

    }

}


// --------------------------------------------------
// 2. DISPLAY DATA PROFILE
// --------------------------------------------------

function showProfile(profile) {

    const profileSection = document.getElementById("profileSection");

    profileSection.style.display = "block";


    // ----------------------------------------------
    // DATA QUALITY STATISTICS
    // ----------------------------------------------

    document.getElementById("stats").innerHTML = `

        <div class="stat">
            <h3>${profile.rows ?? 0}</h3>
            <p>Total Records</p>
        </div>

        <div class="stat">
            <h3>${profile.total_columns ?? 0}</h3>
            <p>Total Columns</p>
        </div>

        <div class="stat">
            <h3>${profile.missing_cells ?? 0}</h3>
            <p>Missing Values</p>
        </div>

        <div class="stat">
            <h3>${profile.duplicate_rows ?? 0}</h3>
            <p>Duplicate Rows</p>
        </div>

    `;


    // ----------------------------------------------
    // DATA SCHEMA TABLE
    // ----------------------------------------------

    document.getElementById("schemaBody").innerHTML =

        (profile.columns || []).map(column => `

            <tr>

                <td>${escapeHTML(column.column)}</td>

                <td>${escapeHTML(column.type)}</td>

                <td>${column.missing ?? 0}</td>

                <td>${column.unique ?? 0}</td>

                <td>
                    <span class="role-badge">
                        ${escapeHTML(column.role)}
                    </span>
                </td>

            </tr>

        `).join("");


    // ----------------------------------------------
    // EXCLUDED AND REFERENCE VARIABLES
    // ----------------------------------------------

    const referenceFields = profile.reference_fields || [];

    const excludedLabels = profile.excluded_labels || [];

    const reviewFields = profile.review_fields || [];

    const excludedCount =
        profile.excluded_count ??
        (referenceFields.length + excludedLabels.length);

    document.getElementById("excludedCount").textContent =
        excludedCount;

    document.getElementById("referenceCount").textContent =
        profile.reference_count ?? referenceFields.length;

    document.getElementById("excludedLabelCount").textContent =
        profile.excluded_label_count ?? excludedLabels.length;


    document.getElementById("referenceFields").textContent =
        referenceFields.length
            ? referenceFields.join(", ")
            : "No reference fields detected";


    document.getElementById("excludedLabels").textContent =
        excludedLabels.length
            ? excludedLabels.join(", ")
            : "No excluded business labels detected";


    document.getElementById("reviewFields").textContent =
        reviewFields.length
            ? reviewFields.join(", ")
            : "No fields requiring review";


    // ----------------------------------------------
    // FEATURE SELECTION
    // ----------------------------------------------

    const featureList = document.getElementById("featureList");

    const features = profile.features || [];

    if (features.length === 0) {

        featureList.innerHTML = `
            <p>No eligible clustering features were detected.</p>
        `;

        return;

    }

    featureList.innerHTML = features.map((feature, index) => {

        const checked = feature.type === "Numeric";

        return `

            <label class="feature-option">

                <input
                    type="checkbox"
                    value="${escapeHTML(feature.column)}"
                    ${checked ? "checked" : ""}
                >

                <span>${escapeHTML(feature.column)}</span>

                <small>${escapeHTML(feature.type)}</small>

            </label>

        `;

    }).join("");


    // Default cluster count
    document.getElementById("clusterCount").value = 3;

}


// --------------------------------------------------
// 3. GET SELECTED FEATURES
// --------------------------------------------------

function getSelectedFeatures() {

    return Array.from(
        document.querySelectorAll(
            "#featureList input[type='checkbox']:checked"
        )
    ).map(input => input.value);

}


// --------------------------------------------------
// 4. RUN CLUSTERING
// --------------------------------------------------

async function runSegmentation() {

    if (!datasetId) {

        alert("Please upload a dataset first.");

        return;

    }

    selectedFeatures = getSelectedFeatures();

    if (selectedFeatures.length === 0) {

        setStatus(
            "profileStatus",
            "Select at least one clustering feature.",
            true
        );

        return;

    }

    const clusterCount = Number(
        document.getElementById("clusterCount").value
    );

    if (
        !Number.isInteger(clusterCount) ||
        clusterCount < 2 ||
        clusterCount > 20
    ) {

        setStatus(
            "profileStatus",
            "Cluster count must be between 2 and 20.",
            true
        );

        return;

    }


    // Initialize default weights for newly selected features
    selectedFeatures.forEach(feature => {

        if (currentWeights[feature] === undefined) {
            currentWeights[feature] = 1;
        }

    });


    const requestBody = {

        dataset_id: datasetId,

        selected_features: selectedFeatures,

        weights: Object.fromEntries(
            selectedFeatures.map(feature => [
                feature,
                Number(currentWeights[feature] ?? 1)
            ])
        ),

        n_clusters: clusterCount

    };


    try {

        setStatus(
            "profileStatus",
            "Running clustering algorithms..."
        );

        setStatus("resultStatus", "");

        const data = await apiRequest(
            `${API}/segment`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(requestBody)
            }
        );


        showResults(data);

        document.getElementById("resultsSection").style.display = "block";

        setStatus(
            "profileStatus",
            "Segmentation completed successfully."
        );

        setStatus(
            "resultStatus",
            "Results generated. You can now explore AI analysis and feature adjustments."
        );

        // Clear previous AI output and suggestions
        const analysis = document.getElementById("aiAnalysis");

        const suggestion = document.getElementById("aiSuggestion");

        if (analysis) analysis.textContent = "";

        if (suggestion) suggestion.textContent = "";

        proposedWeights = {};

        window.scrollTo({
            top: document.getElementById("resultsSection").offsetTop - 30,
            behavior: "smooth"
        });

    } catch (error) {

        setStatus(
            "profileStatus",
            error.message,
            true
        );

    }

}


// --------------------------------------------------
// 5. DISPLAY CLUSTERING RESULTS
// --------------------------------------------------

function showResults(data) {

    const resultsContainer = document.getElementById("results");

    const methods = data.methods || [];

    const bestMethod = data.best_method;


    if (methods.length === 0) {

        resultsContainer.innerHTML = `
            <p>No clustering results were returned.</p>
        `;

        return;

    }


    // ----------------------------------------------
    // ALGORITHM RESULTS
    // ----------------------------------------------

    resultsContainer.innerHTML = `

        <div class="result-summary">

            <h3>Best Performing Algorithm</h3>

            <p>
                ${escapeHTML(bestMethod || "No valid model found")}
            </p>

            <p class="card-description">
                The best algorithm is selected using the
                highest valid silhouette score.
            </p>

        </div>


        <div class="result-grid">

            ${methods.map(method => {

                const isBest = method.method === bestMethod;

                const sizes = method.cluster_sizes || {};

                const clusterDetails = Object.entries(sizes)
                    .map(([name, count]) => `
                        <div>
                            <strong>${escapeHTML(name)}:</strong>
                            ${count} HCPs
                        </div>
                    `)
                    .join("");

                return `

                    <div class="result-card ${isBest ? "best" : ""}">

                        <h3>
                            ${escapeHTML(method.method)}
                            ${isBest ? " ⭐" : ""}
                        </h3>

                        <div class="result-metrics">

                            <span>
                                <strong>Silhouette:</strong>
                                ${method.silhouette ?? "N/A"}
                            </span>

                            <span>
                                <strong>Clusters:</strong>
                                ${method.clusters ?? 0}
                            </span>

                            <span>
                                <strong>Noise:</strong>
                                ${method.noise_points ?? 0}
                            </span>

                        </div>

                        <div class="cluster-details">

                            <h4>Cluster Distribution</h4>

                            ${clusterDetails || "No cluster distribution available"}

                        </div>

                        ${
                            method.error
                            ? `<p class="result-error">${escapeHTML(method.error)}</p>`
                            : ""
                        }

                    </div>

                `;

            }).join("")}

        </div>

    `;


    // ----------------------------------------------
    // ELBOW ANALYSIS
    // ----------------------------------------------

    const elbow = data.elbow_analysis || {};

    if (elbow.tested_values?.length) {

        resultsContainer.innerHTML += `

            <div class="section-heading">

                <h3>Optimal Cluster Analysis</h3>

                <p class="card-description">
                    K-Means was evaluated across different
                    cluster counts.
                </p>

            </div>

            <div class="schema-table-wrapper">

                <table>

                    <thead>

                        <tr>
                            <th>Number of Clusters</th>
                            <th>Inertia</th>
                            <th>Silhouette Score</th>
                        </tr>

                    </thead>

                    <tbody>

                        ${elbow.tested_values.map(item => `

                            <tr>

                                <td>${item.k}</td>

                                <td>${item.inertia}</td>

                                <td>${item.silhouette ?? "N/A"}</td>

                            </tr>

                        `).join("")}

                    </tbody>

                </table>

            </div>

            <p class="selection-note">
                Elbow suggestion:
                ${elbow.suggested_k_by_elbow ?? "Not available"}
                &nbsp; | &nbsp;
                Best silhouette K:
                ${elbow.best_k_by_silhouette ?? "Not available"}
            </p>

        `;

    }


    // ----------------------------------------------
    // CROSS-MODEL COMPARISON
    // ----------------------------------------------

    const comparison = data.assignment_comparison || {};

    if (
        comparison.kmeans_hierarchical_same_partition !== undefined
    ) {

        resultsContainer.innerHTML += `

            <div class="section-heading">

                <h3>Model Agreement</h3>

                <p class="card-description">
                    Comparison of K-Means and Hierarchical
                    cluster membership.
                </p>

                <p>
                    Same partition:
                    <strong>
                        ${
                            comparison.kmeans_hierarchical_same_partition
                                ? "Yes"
                                : "No"
                        }
                    </strong>
                </p>

            </div>

        `;

    }


    // ----------------------------------------------
    // FEATURE WEIGHT SLIDERS
    // ----------------------------------------------

    renderWeights();

}


// --------------------------------------------------
// 6. RENDER FEATURE WEIGHTS
// --------------------------------------------------

function renderWeights() {

    const weightContainer = document.getElementById("weights");

    if (!selectedFeatures.length) {

        weightContainer.innerHTML = "";

        return;

    }

    weightContainer.innerHTML = selectedFeatures.map(feature => {

        const weight = Number(currentWeights[feature] ?? 1);

        return `

            <div class="weight-item">

                <div class="weight-item-header">

                    <strong>${escapeHTML(feature)}</strong>

                    <span id="weightValue-${escapeHTML(feature)}">
                        ${weight.toFixed(1)}
                    </span>

                </div>

                <input
                    type="range"
                    min="0"
                    max="5"
                    step="0.1"
                    value="${weight}"
                    data-feature="${escapeHTML(feature)}"
                    class="feature-weight-slider"
                >

            </div>

        `;

    }).join("");


    document.querySelectorAll(".feature-weight-slider")
        .forEach(slider => {

            slider.addEventListener("input", function () {

                const feature = this.dataset.feature;

                const value = Number(this.value);

                updateWeight(feature, value);

                const valueElement = document.getElementById(
                    `weightValue-${feature}`
                );

                if (valueElement) {
                    valueElement.textContent = value.toFixed(1);
                }

            });

        });

}


// --------------------------------------------------
// 7. UPDATE FEATURE WEIGHT
// --------------------------------------------------

function updateWeight(feature, value) {

    currentWeights[feature] = Number(value);

}


// --------------------------------------------------
// 8. CREATE AI SECTION
// --------------------------------------------------

function addAISection() {

    const resultsSection = document.getElementById("resultsSection");

    if (document.getElementById("aiSection")) return;

    const section = document.createElement("div");

    section.id = "aiSection";

    section.className = "ai-panel";

    section.innerHTML = `

        <div class="section-label">STEP 04</div>

        <h2>AI-Powered Segmentation Insights</h2>

        <p class="card-description">
            Use  AI to interpret clustering results
            and explore possible feature-weight adjustments.
        </p>


        <div class="ai-buttons">

            <button
                type="button"
                onclick="runAIAnalysis()"
            >
                Generate AI Analysis
            </button>

        </div>

        <div
            id="aiAnalysis"
            class="ai-output"
        ></div>


        <div class="section-heading">

            <h3>Natural Language Feature Adjustment</h3>

            <p class="card-description">
                Describe how you want the segmentation to change.
                AI will suggest updated feature weights.
            </p>

        </div>


        <textarea
            id="aiInstruction"
            placeholder="Example: Give more importance to prescription volume and engagement score."
        ></textarea>


        <div class="ai-buttons">

            <button
                type="button"
                onclick="suggestChanges()"
            >
                Generate Weight Suggestions
            </button>

            <button
                type="button"
                onclick="approveChanges()"
            >
                Apply Suggested Weights & Re-run
            </button>

        </div>


        <div
            id="aiSuggestion"
            class="ai-output"
        ></div>

    `;

    resultsSection.appendChild(section);

}


// --------------------------------------------------
// 9. RUN AI ANALYSIS
// --------------------------------------------------

async function runAIAnalysis() {

    if (!datasetId) return;

    const output = document.getElementById("aiAnalysis");

    output.textContent = "Generating AI analysis...";

    try {

        const data = await apiRequest(
            `${API}/analyze/${datasetId}`
        );

        output.textContent =
            typeof data.analysis === "string"
                ? data.analysis
                : JSON.stringify(data.analysis, null, 2);

    } catch (error) {

        output.textContent = `AI analysis failed: ${error.message}`;

    }

}


// --------------------------------------------------
// 10. SUGGEST FEATURE WEIGHT CHANGES
// --------------------------------------------------

async function suggestChanges() {

    if (!datasetId) return;

    const instruction = document.getElementById(
        "aiInstruction"
    ).value.trim();

    const output = document.getElementById("aiSuggestion");

    if (instruction.length < 3) {

        output.textContent =
            "Please enter an instruction describing the changes you want.";

        return;

    }

    if (!selectedFeatures.length) {

        output.textContent =
            "Please run segmentation first.";

        return;

    }

    output.textContent = "Generating feature-weight suggestions...";

    try {

        const data = await apiRequest(
            `${API}/suggest-adjustments`,
            {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({

                    dataset_id: datasetId,

                    instruction: instruction,

                    selected_features: selectedFeatures,

                    current_weights: currentWeights

                })

            }
        );


        proposedWeights = data.proposed_weights || {};


        output.textContent =

            (data.explanation || "AI suggestions generated.") +

            "\n\nProposed Feature Weights:\n" +

            Object.entries(proposedWeights)
                .map(([feature, weight]) =>
                    `${feature}: ${weight}`
                )
                .join("\n");


    } catch (error) {

        output.textContent =
            `Could not generate suggestions: ${error.message}`;

    }

}


// --------------------------------------------------
// 11. APPLY AI-SUGGESTED WEIGHTS
// --------------------------------------------------

async function approveChanges() {

    if (!Object.keys(proposedWeights).length) {

        setStatus(
            "resultStatus",
            "Generate AI weight suggestions first.",
            true
        );

        return;

    }


    // Apply only suggestions for selected features
    selectedFeatures.forEach(feature => {

        if (proposedWeights[feature] !== undefined) {

            const weight = Number(proposedWeights[feature]);

            if (Number.isFinite(weight)) {

                currentWeights[feature] = Math.min(
                    5,
                    Math.max(0, weight)
                );

            }

        }

    });


    // Update sliders visually
    renderWeights();


    // Run clustering with the updated weights
    await runSegmentation();

}


// --------------------------------------------------
// 12. DOWNLOAD EXCEL REPORT
// --------------------------------------------------

async function downloadReport() {

    if (!datasetId) {

        alert("Please run segmentation first.");

        return;

    }

    try {

        setStatus(
            "resultStatus",
            "Preparing Excel report..."
        );

        const response = await fetch(
            `${API}/report/${datasetId}`
        );

        if (!response.ok) {

            let errorData = {};

            try {
                errorData = await response.json();
            } catch {}

            throw new Error(
                errorData.detail || "Could not download report."
            );

        }

        const blob = await response.blob();

        const url = window.URL.createObjectURL(blob);

        const link = document.createElement("a");

        link.href = url;

        link.download = "hcp_segmentation_report.xlsx";

        document.body.appendChild(link);

        link.click();

        link.remove();

        window.URL.revokeObjectURL(url);


        setStatus(
            "resultStatus",
            "Excel report downloaded successfully."
        );

    } catch (error) {

        setStatus(
            "resultStatus",
            error.message,
            true
        );

    }

}


// --------------------------------------------------
// 13. INITIALIZE
// --------------------------------------------------

document.addEventListener("DOMContentLoaded", function () {

    addAISection();

});