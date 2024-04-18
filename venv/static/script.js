const themeToggle = document.getElementById('themeToggle');

themeToggle.addEventListener('change', () => {
    document.body.classList.toggle('night-theme');
});
