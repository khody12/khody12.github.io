document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. LIGHTBOX FUNCTIONALITY ---
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox';
    lightbox.innerHTML = `<img src="" alt="Full screen view">`;
    document.body.appendChild(lightbox);
    
    const lightboxImg = lightbox.querySelector('img');
    
    // Select all images inside gallery items to be zoomable
    const images = document.querySelectorAll('.gallery-item img');
    
    images.forEach(img => {
        img.addEventListener('click', () => {
            lightboxImg.src = img.src;
            lightbox.classList.add('active');
        });
    });
    
    // Click outside image to close
    lightbox.addEventListener('click', (e) => {
        if (e.target !== lightboxImg) {
            lightbox.classList.remove('active');
        }
    });

    // --- 2. BEFORE/AFTER SLIDER LOGIC ---
    const sliders = document.querySelectorAll('.comp-container');
    
    sliders.forEach(slider => {
        const overlay = slider.querySelector('.comp-overlay');
        let isDown = false;

        // Mouse events
        slider.addEventListener('mousedown', () => isDown = true);
        window.addEventListener('mouseup', () => isDown = false);
        
        slider.addEventListener('mousemove', (e) => {
            if (!isDown) return;
            handleMove(e, slider, overlay);
        });

        // Touch events for mobile
        slider.addEventListener('touchstart', () => isDown = true);
        window.addEventListener('touchend', () => isDown = false);
        
        slider.addEventListener('touchmove', (e) => {
            if (!isDown) return;
            handleMove(e.touches[0], slider, overlay);
        });
        
        // Handle click to jump
        slider.addEventListener('click', (e) => {
            handleMove(e, slider, overlay);
        });
    });

    function handleMove(e, slider, overlay) {
        const rect = slider.getBoundingClientRect();
        let x = e.clientX - rect.left;
        
        // Constrain x within the slider
        if (x < 0) x = 0;
        if (x > rect.width) x = rect.width;
        
        const percentage = (x / rect.width) * 100;
        overlay.style.width = `${percentage}%`;
    }
});