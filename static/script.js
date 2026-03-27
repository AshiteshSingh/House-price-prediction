/**
 * House Price Prediction - Frontend Controller
 *
 * Handles form submission, input validation, API communication,
 * and animated result/error display.
 */

(function () {
    "use strict";

    // ---------------------------------------------------------------
    // DOM References
    // ---------------------------------------------------------------

    const form = document.getElementById("predict-form");
    const btn = document.getElementById("predict-btn");
    const resultPanel = document.getElementById("result-panel");
    const resultPrice = document.getElementById("result-price");
    const resultMeta = document.getElementById("result-meta");
    const errorPanel = document.getElementById("error-panel");

    // ---------------------------------------------------------------
    // Event Binding
    // ---------------------------------------------------------------

    form.addEventListener("submit", handleSubmit);

    // ---------------------------------------------------------------
    // Handlers
    // ---------------------------------------------------------------

    async function handleSubmit(event) {
        event.preventDefault();
        hideResults();

        const payload = parseFormValues();
        if (!payload) return;

        setLoading(true);

        try {
            const data = await fetchPrediction(payload);
            displayResult(data);
        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(false);
        }
    }

    // ---------------------------------------------------------------
    // Form Parsing & Validation
    // ---------------------------------------------------------------

    function parseFormValues() {
        const formData = new FormData(form);
        const payload = {};

        for (const [key, value] of formData.entries()) {
            const trimmed = value.trim();
            if (trimmed === "") {
                showError("Please fill in all fields.");
                return null;
            }
            const num = parseFloat(trimmed);
            if (isNaN(num)) {
                showError("Invalid number for " + key + ".");
                return null;
            }
            payload[key] = num;
        }

        return payload;
    }

    // ---------------------------------------------------------------
    // API Communication
    // ---------------------------------------------------------------

    async function fetchPrediction(payload) {
        const response = await fetch("https://jarvisai1234-house-price-prediction-india.hf.space/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction request failed.");
        }

        return data;
    }

    // ---------------------------------------------------------------
    // UI Updates
    // ---------------------------------------------------------------

    function displayResult(data) {
        resultPrice.textContent = data.price_formatted;
        resultMeta.innerHTML =
            "Model: <span>" + data.model_used + "</span>" +
            " &middot; R&sup2; Score: <span>" + data.r2_score + "</span>";

        requestAnimationFrame(function () {
            resultPanel.classList.add("visible");
        });
    }

    function showError(message) {
        errorPanel.textContent = message;
        errorPanel.classList.add("visible");
        setTimeout(function () {
            errorPanel.classList.remove("visible");
        }, 5000);
    }

    function hideResults() {
        resultPanel.classList.remove("visible");
        errorPanel.classList.remove("visible");
    }

    function setLoading(isLoading) {
        if (isLoading) {
            btn.classList.add("loading");
            btn.disabled = true;
        } else {
            btn.classList.remove("loading");
            btn.disabled = false;
        }
    }
})();
