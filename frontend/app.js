// Initialize Lucide icons
lucide.createIcons();

document.addEventListener('DOMContentLoaded', () => {
    // Theme toggling logic
    const themeToggle = document.getElementById('theme-toggle');
    const htmlElement = document.documentElement;
    const themeIcon = document.getElementById('theme-icon');

    // Default to dark mode based on HTML class
    let isDark = htmlElement.classList.contains('dark');

    themeToggle.addEventListener('click', () => {
        isDark = !isDark;
        if (isDark) {
            htmlElement.classList.add('dark');
            themeIcon.setAttribute('data-lucide', 'moon');
        } else {
            htmlElement.classList.remove('dark');
            themeIcon.setAttribute('data-lucide', 'sun');
        }
        lucide.createIcons(); // Re-render icon
    });

    const elements = {
        stateBadge: document.getElementById('state-badge'),
        dynLabel: document.getElementById('dyn-label'),
        dynConf: document.getElementById('dyn-conf'),
        statLabel: document.getElementById('stat-label'),
        statConf: document.getElementById('stat-conf'),
        wristEnergy: document.getElementById('wrist-energy'),
        wristBar: document.getElementById('wrist-bar'),
        angleEnergy: document.getElementById('angle-energy'),
        angleBar: document.getElementById('angle-bar'),
        wordBuffer: document.getElementById('word-buffer'),
        timeoutBar: document.getElementById('timeout-bar'),
        sentenceHistory: document.getElementById('sentence-history'),
        emptyHistory: document.getElementById('empty-history'),
        btnClear: document.getElementById('btn-clear'),
        btnForce: document.getElementById('btn-force'),
        cameraStatus: document.getElementById('camera-status'),
        modelMode: document.getElementById('model-mode')
    };

    const PAUSE_FRAMES = 25;
    let lastHistoryJson = "";
    
    // Update UI Function from Server State
    function updateDashboard(data) {
        // State badge
        if (!data.handsDetected) {
            elements.stateBadge.textContent = 'NO HANDS';
            elements.stateBadge.className = 'badge badge-outline';
            elements.stateBadge.style.color = 'hsl(var(--muted-foreground))';
            elements.cameraStatus.classList.remove('active');
            elements.cameraStatus.style.backgroundColor = '#ef4444';
        } else if (data.moving) {
            elements.stateBadge.textContent = 'SIGNING';
            elements.stateBadge.className = 'badge badge-primary';
            elements.stateBadge.style.backgroundColor = '#10b981';
            elements.stateBadge.style.color = 'white';
            elements.cameraStatus.classList.add('active');
            elements.cameraStatus.style.backgroundColor = '#10b981';
        } else {
            elements.stateBadge.textContent = 'STILL';
            elements.stateBadge.className = 'badge badge-secondary';
            elements.stateBadge.style.backgroundColor = '#3b82f6';
            elements.stateBadge.style.color = 'white';
            elements.cameraStatus.classList.add('active');
            elements.cameraStatus.style.backgroundColor = '#10b981';
        }

        // Metrics
        elements.wristEnergy.textContent = data.wristE.toFixed(4);
        elements.angleEnergy.textContent = data.angleE.toFixed(4);
        
        const wristPct = Math.min((data.wristE / 0.01) * 100, 100);
        const anglePct = Math.min((data.angleE / 0.01) * 100, 100);
        
        elements.wristBar.style.width = `${wristPct}%`;
        elements.wristBar.style.backgroundColor = data.wristE > 0.006 ? '#10b981' : 'hsl(var(--primary))';
        
        elements.angleBar.style.width = `${anglePct}%`;
        elements.angleBar.style.backgroundColor = data.angleE > 0.003 ? '#10b981' : 'hsl(var(--primary))';

        // Outputs
        if (data.dynOutput.label) {
            elements.dynLabel.textContent = data.dynOutput.label;
            elements.dynConf.textContent = data.dynOutput.conf.toFixed(2);
            elements.dynConf.style.backgroundColor = data.dynOutput.conf > 0.6 ? '#10b981' : 'hsl(var(--secondary))';
            elements.dynConf.style.color = data.dynOutput.conf > 0.6 ? 'white' : 'inherit';
        } else {
            elements.dynLabel.textContent = '-';
            elements.dynConf.textContent = '0.00';
            elements.dynConf.style.backgroundColor = 'hsl(var(--secondary))';
            elements.dynConf.style.color = 'inherit';
        }

        if (data.statOutput.label) {
            elements.statLabel.textContent = data.statOutput.label;
            elements.statConf.textContent = data.statOutput.conf.toFixed(2);
            elements.statConf.style.backgroundColor = data.statOutput.conf > 0.85 ? '#a855f7' : 'hsl(var(--secondary))';
            elements.statConf.style.color = data.statOutput.conf > 0.85 ? 'white' : 'inherit';
        } else {
            elements.statLabel.textContent = '-';
            elements.statConf.textContent = '0.00';
            elements.statConf.style.backgroundColor = 'hsl(var(--secondary))';
            elements.statConf.style.color = 'inherit';
        }

        // Mode Indicator
        if (data.moving && data.handsDetected) {
            elements.modelMode.innerHTML = '<i data-lucide="zap" style="width: 12px; height: 12px; margin-right: 4px;"></i>Dynamic Mode';
            elements.modelMode.style.backgroundColor = '#10b981';
        } else if (!data.moving && data.handsDetected) {
            elements.modelMode.innerHTML = '<i data-lucide="anchor" style="width: 12px; height: 12px; margin-right: 4px;"></i>Static Mode';
            elements.modelMode.style.backgroundColor = '#a855f7';
        }

        // Render Words Buffer
        if (data.words.length === 0) {
            elements.wordBuffer.innerHTML = '<span class="text-sm text-muted">No signs detected yet...</span>';
        } else {
            elements.wordBuffer.innerHTML = data.words.map(w => `<span class="word-pill">${w}</span>`).join('');
        }

        // Render Timeout Bar
        if (data.words.length > 0 && data.noHandFrames > 0) {
            const progress = Math.min((data.noHandFrames / PAUSE_FRAMES) * 100, 100);
            elements.timeoutBar.style.width = `${progress}%`;
        } else {
            elements.timeoutBar.style.width = '0%';
        }

        // Render History (only if changed to prevent animation flickering)
        const currentHistoryJson = JSON.stringify(data.history);
        if (currentHistoryJson !== lastHistoryJson) {
            lastHistoryJson = currentHistoryJson;
            if (data.history.length === 0) {
                elements.emptyHistory.style.display = 'flex';
                const blocks = elements.sentenceHistory.querySelectorAll('.sentence-bubble');
                blocks.forEach(b => b.remove());
            } else {
                elements.emptyHistory.style.display = 'none';
                // Simple render: clear all bubbles and recreate them
                const blocks = elements.sentenceHistory.querySelectorAll('.sentence-bubble');
                blocks.forEach(b => b.remove());
                
                // Reverse history to show newest at the top
                [...data.history].reverse().forEach(entry => {
                    const bubble = document.createElement('div');
                    bubble.className = 'sentence-bubble';
                    bubble.innerHTML = `
                        <div class="flex items-start gap-3">
                            <i data-lucide="bot" style="margin-top: 2px; color: hsl(var(--muted-foreground));"></i>
                            <div>
                                <p class="font-medium text-sm text-muted mb-1">Translated from: <span style="font-weight: normal">${entry.words.join(', ')}</span></p>
                                <p class="text-foreground">${entry.sentence}</p>
                            </div>
                        </div>
                    `;
                    elements.sentenceHistory.appendChild(bubble);
                });
            }
            lucide.createIcons();
        }
    }

    // Connect to Server-Sent Events
    const source = new EventSource('/state');
    source.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };

    // Button event listeners
    elements.btnClear.addEventListener('click', () => {
        fetch('/clear', { method: 'POST' });
    });

    elements.btnForce.addEventListener('click', () => {
        fetch('/force_translate', { method: 'POST' });
    });

    const modeButtons = document.querySelectorAll('.mode-toggle');
    modeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.getAttribute('data-mode');
            fetch('/set_mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode })
            });
            // Update UI optimistically
            modeButtons.forEach(b => {
                b.style.background = 'transparent';
                b.style.color = 'hsl(var(--muted-foreground))';
                b.style.boxShadow = 'none';
                b.classList.remove('active');
            });
            btn.style.background = 'hsl(var(--background))';
            btn.style.color = 'hsl(var(--foreground))';
            btn.style.boxShadow = '0 1px 2px rgba(0,0,0,0.1)';
            btn.classList.add('active');
        });
    });
});
