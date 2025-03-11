document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("uploadForm").addEventListener("submit", async function (event) {
        event.preventDefault();
        const fileInput = document.getElementById("imageInput");
        const file = fileInput.files[0];
        const preview = document.getElementById("preview");
        const outputImage = document.getElementById("outputImage");
        const resultsDiv = document.getElementById("results");

        if (!file) {
            resultsDiv.innerHTML = '<div class="alert alert-danger">Please select an image.</div>';
            return;
        }

        // Show uploaded image preview
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";

        // Clear previous results
        resultsDiv.innerHTML = `<div class="alert alert-info">Processing image...</div>`;

        // Prepare FormData
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error("Server error:", errorText);
                resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${errorText}</div>`;
                return;
            }

            const data = await response.json();
            console.log("Response received:", data);

            outputImage.src = data.image_url;
            outputImage.style.display = "block";

            resultsDiv.innerHTML = `
                <h4 class="mt-4">Parking Lot Stats:</h4>
                <p><strong>Empty Spots:</strong> ${data.empty_spots}</p>
                <p><strong>Occupied Spots:</strong> ${data.occupied_spots}</p>
                <a href="${data.text_file_url}" class="btn btn-success mt-3" download>Download Detection Results (.txt)</a>
            `;

        } catch (error) {
            console.error("Error:", error);
            resultsDiv.innerHTML = '<div class="alert alert-danger">An error occurred while processing the image.</div>';
        }
    });
});
