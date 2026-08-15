// Wait until page loads
document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");
    const button = document.querySelector("button");

    form.addEventListener("submit", function () {

        button.innerHTML = "Checking...";
        button.disabled = true;

    });

});