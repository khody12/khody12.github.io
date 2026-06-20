/* js/main.js */
document.addEventListener('DOMContentLoaded', () => {
    
    // --- LIGHTBOX ---
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.innerHTML = `<img src="" alt="Full view">`;
    document.body.appendChild(lightbox);
    
    const lightboxImg = lightbox.querySelector('img');
    
    // Select all images in gallery items
    const images = document.querySelectorAll('.gallery-item img');
    images.forEach(img => {
        img.addEventListener('click', () => {
            lightboxImg.src = img.src;
            lightbox.classList.add('active');
        });
    });
    
    lightbox.addEventListener('click', () => lightbox.classList.remove('active'));

    // --- SLIDERS ---
    const sliders = document.querySelectorAll('.comp-container');
    sliders.forEach(slider => {
        const overlay = slider.querySelector('.comp-overlay');
        const bgImg = slider.querySelector('.comp-img');
        const overlayImg = overlay.querySelector('img');

        // Sync overlay image width to background image width to prevent squishing
        function syncWidth() {
            overlayImg.style.width = `${bgImg.offsetWidth}px`;
        }
        window.addEventListener('resize', syncWidth);
        // Run once on load (and slight delay for image loading)
        if (bgImg.complete) syncWidth();
        else bgImg.onload = syncWidth;

        let isDown = false;
        
        function move(e) {
            if (!isDown) return;
            const rect = slider.getBoundingClientRect();
            let x = (e.pageX || e.touches[0].pageX) - rect.left - window.scrollX;
            if (x < 0) x = 0;
            if (x > rect.width) x = rect.width;
            overlay.style.width = `${x}px`;
        }

        slider.addEventListener('mousedown', () => isDown = true);
        slider.addEventListener('touchstart', () => isDown = true);
        
        window.addEventListener('mouseup', () => isDown = false);
        window.addEventListener('touchend', () => isDown = false);
        
        slider.addEventListener('mousemove', move);
        slider.addEventListener('touchmove', move);
    });
});