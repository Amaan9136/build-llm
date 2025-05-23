// Main JavaScript for LLM Training Platform

document.addEventListener('DOMContentLoaded', function() {
    // Add active class to current nav item
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (currentPath === linkPath) {
            link.classList.add('active');
        } else if (currentPath.includes(linkPath) && linkPath !== '/') {
            link.classList.add('active');
        }
    });
    
    // Flash message handling
    const flashMessages = document.querySelectorAll('.alert-dismissible');
    flashMessages.forEach(flash => {
        setTimeout(() => {
            // Add fade out class
            flash.classList.add('fade');
            
            // Remove after animation completes
            setTimeout(() => {
                flash.remove();
            }, 500);
        }, 5000);
    });
});