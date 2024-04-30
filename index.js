

function validateForm() {
    var fileInput = document.getElementById('uploadedImage');
    
    // Check if a file is selected
    if (fileInput.files.length === 0) {
        alert("Please select a file.");
        return false; // Prevent form submission
    }

    return true; // Allow form submission
}
