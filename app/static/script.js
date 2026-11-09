
/* =========================================
   CHURNGUARD - FRONTEND LOGIC
========================================= */

// DOM Elements

const predictionForm = document.getElementById("prediction-form");
const predictButton = document.getElementById("predict-btn");

const inputSection = document.getElementById("input-section");
const resultsSection = document.getElementById("results-section");

const predictionResults = document.getElementById("prediction-results");
const errorMessage = document.getElementById("error-message");

const newPredictionButton = document.getElementById("new-prediction-btn");

const metricsTableContainer = document.getElementById(
    "metrics-table-container"
);

let modelChart = null;


// =========================================
// 1. COLLECT CUSTOMER INPUT
// =========================================

function getCustomerData() {

    return {
        credit_score: Number(document.getElementById("credit_score").value),

        country: document.getElementById("country").value,

        gender: document.getElementById("gender").value,

        age: Number(document.getElementById("age").value),

        tenure: Number(document.getElementById("tenure").value),

        balance: Number(document.getElementById("balance").value),

        products_number: Number(
            document.getElementById("products_number").value
        ),

        credit_card: Number(
            document.getElementById("credit_card").value
        ),

        active_member: Number(
            document.getElementById("active_member").value
        ),

        estimated_salary: Number(
            document.getElementById("estimated_salary").value
        )
    };
}


// =========================================
// 2. SHOW ERROR MESSAGE
// =========================================

function showError(message) {

    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}


function hideError() {

    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}


// =========================================
// 3. LOADING STATE
// =========================================

function setLoading(isLoading) {

    predictButton.disabled = isLoading;

    predictButton.innerHTML = isLoading
        ? "Analyzing Customer..."
        : 'Predict Churn <span>→</span>';
}


// =========================================
// 4. SUBMIT PREDICTION REQUEST
// =========================================

predictionForm.addEventListener("submit", async function (event) {

    event.preventDefault();

    hideError();

    if (!predictionForm.reportValidity()) {
        return;
    }

    const customerData = getCustomerData();

    setLoading(true);

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(customerData)

        });


        if (!response.ok) {

            const errorData = await response.json();

            throw new Error(
                errorData.detail || "Prediction request failed."
            );
        }


        const predictions = await response.json();


        // Display prediction results

        displayPredictions(predictions);


        // Load model comparison metrics

        await loadModelMetrics();


        // Switch to results section

        inputSection.classList.add("hidden");
        resultsSection.classList.remove("hidden");

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }

    catch (error) {

        console.error("Prediction Error:", error);

        showError(
            error.message || "Unable to connect to prediction server."
        );

    }

    finally {

        setLoading(false);

    }

});


// =========================================
// 5. DISPLAY THREE MODEL PREDICTIONS
// =========================================

function displayPredictions(predictions) {

    predictionResults.innerHTML = "";

    const models = [
        {
            key: "logistic_regression",
            name: "Logistic Regression"
        },
        {
            key: "random_forest",
            name: "Random Forest"
        },
        {
            key: "xgboost",
            name: "XGBoost"
        }
    ];


    models.forEach(model => {

        const result = predictions[model.key];

        if (!result) {
            throw new Error(`Missing prediction for ${model.name}.`);
        }

        const probability = Number(result.churn_probability) * 100;

        const isHighRisk = Number(result.prediction) === 1;

        const riskClass = isHighRisk ? "high" : "low";

        const riskLabel = isHighRisk
            ? "Likely to Churn"
            : "Unlikely to Churn";

        const card = document.createElement("div");

        card.className = "prediction-card";

        card.innerHTML = `

            <h3>${model.name}</h3>

            <div class="prediction-status ${riskClass}">
                ${riskLabel}
            </div>

            <p class="prediction-probability">
                Churn Probability:
                <strong>${probability.toFixed(2)}%</strong>
            </p>

            <div class="probability-bar">

                <div
                    class="probability-fill"
                    style="width: ${Math.min(100, Math.max(0, probability))}%">
                </div>

            </div>

        `;

        predictionResults.appendChild(card);

    });

}


// =========================================
// 6. FETCH MODEL EVALUATION METRICS
// =========================================

async function loadModelMetrics() {

    try {

        const response = await fetch("/models/metrics");

        if (!response.ok) {
            throw new Error("Metrics request failed (" + response.status + ")");
        }

        const raw = await response.json();
        console.log("Raw metrics:", raw);

        const norm = s => String(s).toLowerCase().replace(/[^a-z]/g, "");
        const keyMap = {
            logisticregression: "logistic_regression",
            randomforest: "random_forest",
            xgboost: "xgboost"
        };

        let src = raw.metrics ?? raw.models ?? raw.results ?? raw;
        const metrics = {};

        const add = (name, v) => {
            const key = keyMap[norm(name)];
            if (!key) return;
            const pick = (...names) => {
                for (const n of names) {
                    for (const k of Object.keys(v)) {
                        if (norm(k) === norm(n)) return v[k];
                    }
                }
                return undefined;
            };
            metrics[key] = {
                roc_auc: pick("roc_auc", "auc", "rocauc"),
                accuracy: pick("accuracy"),
                precision: pick("precision"),
                recall: pick("recall"),
                f1: pick("f1", "f1_score", "f1score")
            };
        };

        if (Array.isArray(src)) {
            src.forEach(i => add(i.model ?? i.model_name ?? i.name, i));
        } else {
            Object.entries(src).forEach(([k, v]) => add(k, v));
        }

        displayMetricsTable(metrics);
        displayModelChart(metrics);

    } catch (error) {

        console.error("Metrics Error:", error);

        metricsTableContainer.innerHTML = `
            <p class="error-message">
                Model comparison metrics could not be loaded:
                ${error.message}
            </p>
        `;
    }
}

// =========================================
// 7. DISPLAY MODEL METRICS TABLE
// =========================================

function displayMetricsTable(metrics) {

    const modelNames = {
        logistic_regression: "Logistic Regression",
        random_forest: "Random Forest",
        xgboost: "XGBoost"
    };

    const metricNames = [
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ];

    let tableHTML = `

        <table class="metrics-table">

            <thead>
                <tr>
                    <th>Model</th>
                    <th>ROC-AUC</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                </tr>
            </thead>

            <tbody>
    `;


    Object.keys(modelNames).forEach(modelKey => {

        const modelMetrics = metrics[modelKey];

        if (!modelMetrics) {
            throw new Error(`Missing metrics for ${modelKey}`);
        }

        tableHTML += `
            <tr>
                <td>${modelNames[modelKey]}</td>

                ${metricNames.map(metric => {

                    const value = Number(modelMetrics[metric]);

                    if (!Number.isFinite(value)) {
                        throw new Error(`Invalid metric: ${metric}`);
                    }

                    return `<td>${value.toFixed(4)}</td>`;

                }).join("")}

            </tr>
        `;

    });


    tableHTML += `

            </tbody>

        </table>

    `;


    metricsTableContainer.innerHTML = tableHTML;

}


// =========================================
// 8. MODEL COMPARISON CHART
// =========================================

function displayModelChart(metrics) {

    const canvas = document.getElementById("model-chart");

    const context = canvas.getContext("2d");

    const modelKeys = [
        "logistic_regression",
        "random_forest",
        "xgboost"
    ];

    const modelNames = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ];

    const rocAucValues = modelKeys.map(key =>
        Number(metrics[key].roc_auc)
    );


    // Destroy previous chart if it exists

    if (modelChart) {
        modelChart.destroy();
    }


    modelChart = new Chart(context, {

        type: "bar",

        data: {

            labels: modelNames,

            datasets: [{

                label: "ROC-AUC",

                data: rocAucValues,

                backgroundColor: [
                    "#687443",
                    "#87945B",
                    "#B4C18A"
                ],

                borderRadius: 7,

                maxBarThickness: 65

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {
                    display: false
                },

                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return "ROC-AUC: " +
                                context.parsed.y.toFixed(4);
                        }
                    }
                }

            },

            scales: {

                y: {

                    beginAtZero: false,

                    min: 0,

                    max: 1,

                    ticks: {
                        color: "#A0A494"
                    },

                    grid: {
                        color: "#34382C"
                    }

                },

                x: {

                    ticks: {
                        color: "#D5D8CB"
                    },

                    grid: {
                        display: false
                    }

                }

            }

        }

    });

}


// =========================================
// 9. NEW PREDICTION
// =========================================

newPredictionButton.addEventListener("click", function () {

    // Reset previous form data

    predictionForm.reset();

    // Clear previous results

    predictionResults.innerHTML = "";

    metricsTableContainer.innerHTML = "";

    hideError();


    // Destroy previous chart

    if (modelChart) {

        modelChart.destroy();

        modelChart = null;

    }


    // Switch back to input section

    resultsSection.classList.add("hidden");

    inputSection.classList.remove("hidden");


    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });

});


// =========================================
// 10. INITIALIZATION
// =========================================

console.log("ChurnGuard frontend initialized successfully.");
