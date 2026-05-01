document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("resumeFile");
    const jdInput = document.getElementById("jdText");
    const submitBtn = document.getElementById("submitBtn");
    const errorMessage = document.getElementById("errorMessage");
    
    // UI Sections
    const welcomeScreen = document.getElementById("welcomeScreen");
    const loader = document.getElementById("loader");
    const resultsDashboard = document.getElementById("resultsDashboard");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // 1. Validate Input
        const file = fileInput.files[0];
        if (!file) {
            showError("Please select a PDF file.");
            return;
        }

        // 2. Prepare UI for loading
        hideError();
        welcomeScreen.classList.add("hidden");
        resultsDashboard.classList.add("hidden");
        loader.classList.remove("hidden");
        submitBtn.disabled = true;
        submitBtn.innerText = "🔍 Analyzing...";

        // 3. Prepare Data for API
        const formData = new FormData();
        formData.append("file", file);
        formData.append("jd", jdInput.value);

        // 4. Send request to FastAPI Backend
        try {
            // Make sure your FastAPI server is running on port 8000!
            const response = await fetch("http://localhost:8000/api/analyze", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                renderDashboard(result.data);
            } else {
                showError(result.error || "An unknown error occurred on the server.");
                resetUI();
            }

        } catch (error) {
            console.error("Fetch error:", error);
            showError("Failed to connect to the backend. Is FastAPI running on localhost:8000?");
            resetUI();
        }
    });

    function renderDashboard(data) {
        // Update Metrics
        document.getElementById("scoreOverall").innerText = `${data.score}/100`;
        document.getElementById("scoreAts").innerText = data.ats.score !== null ? `${data.ats.score}%` : "—";
        document.getElementById("scoreImpact").innerText = `${data.impact_pct}%`;
        document.getElementById("scoreSections").innerText = `${data.sections.found.length}/8`;

        // Render Action Plan
        const planList = document.getElementById("actionPlanList");
        planList.innerHTML = ""; // Clear old items

        data.action_plan.forEach(item => {
            // item is expected to be an array like ["🔴 Critical", "Message..."]
            const priorityLabel = item[0];
            const message = item[1];
            
            // Map the emoji label to a CSS class
            let cssClass = "priority-medium";
            if (priorityLabel.includes("Critical")) cssClass = "priority-critical";
            if (priorityLabel.includes("High")) cssClass = "priority-high";
            if (priorityLabel.includes("Great")) cssClass = "priority-great";

            const li = document.createElement("li");
            li.className = cssClass;
            li.innerHTML = `<strong>${priorityLabel}</strong>: ${message}`;
            planList.appendChild(li);
        });

        // Show Dashboard
        loader.classList.add("hidden");
        resultsDashboard.classList.remove("hidden");
        
        // Reset Button
        submitBtn.disabled = false;
        submitBtn.innerText = "🔍 Analyze Resume";
    }

    function showError(msg) {
        errorMessage.innerText = msg;
        errorMessage.classList.remove("hidden");
    }

    function hideError() {
        errorMessage.classList.add("hidden");
    }

    function resetUI() {
        loader.classList.add("hidden");
        welcomeScreen.classList.remove("hidden");
        submitBtn.disabled = false;
        submitBtn.innerText = "🔍 Analyze Resume";
    }
});
