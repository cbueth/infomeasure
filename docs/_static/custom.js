document.addEventListener('DOMContentLoaded', function() {
    if (document.title.includes('Overview')) {
        document.querySelector('.sidebar-toggle.primary-toggle.btn.btn-sm').style.display = 'none';
    }
});
