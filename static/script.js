document.getElementById("predictForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById("predictionResult");
        const predictionText = document.getElementById("prediction");
        resultDiv.style.display = "block";
        predictionText.textContent = `Cattle Weight: ${data.prediction[0].toFixed(2)} KG`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

document.getElementById("uploadForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById("uploadResult");
        const uploadMessage = document.getElementById("uploadMessage");
        const uploadedImage = document.getElementById("uploadedImage");

        resultDiv.style.display = "block";
        uploadMessage.textContent = `Cattle Weight: ${data.prediction[0].toFixed(2)} KG`;
        uploadedImage.src = data.image_url;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

function showPage(page) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(p => p.style.display = 'none');
    document.getElementById(page).style.display = 'block';
}
