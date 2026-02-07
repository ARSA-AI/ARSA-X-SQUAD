/**
 * ARSA X SQUAD - Physics + AI Fusion Animation
 * Auto-starts on page load
 */

// =============================================
// PHYSICS + AI ANIMATION (Optical Flow + Neural Net)
// =============================================

// =============================================
// PHYSICS + AI ANIMATION (Professional Neuro-Symbolic Fusion)
// =============================================

// =============================================
// TRUE PHYSICS-BASED NEURO-SYMBOLIC CORE
// =============================================

const canvas = document.getElementById('physics-canvas');
const ctx = canvas.getContext('2d');

// -- Configuration --
const NODE_COUNT = 40;
const CONNECTION_DIST = 100;
const MOUSE_INFLUENCE = 150;
const SPRING_STRENGTH = 0.05;
const DAMPING = 0.98;

// Match Theme Colors
const COLOR_ACCENT = '41, 151, 255'; // #2997ff
const COLOR_PURPLE = '191, 90, 242'; // #bf5af2

// High DPI Resize
let width, height;
function resize() {
    const parent = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    width = parent.clientWidth;
    height = 500;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';

    ctx.scale(dpr, dpr);
}
window.addEventListener('resize', resize);
resize();

// -- Physics Engine Components --

class Node {
    constructor(x, y, fixed = false) {
        this.x = x;
        this.y = y;
        this.vx = (Math.random() - 0.5) * 2;
        this.vy = (Math.random() - 0.5) * 2;
        this.mass = 1 + Math.random() * 2;
        this.radius = Math.random() < 0.2 ? 6 : 3; // Some "Bias" nodes are larger
        this.fixed = fixed; // Input/Output nodes might be tethered
        this.energy = 0; // Activation level
    }

    update() {
        if (this.fixed) return;

        // Apply Damping (Friction)
        this.vx *= DAMPING;
        this.vy *= DAMPING;

        // Update Position
        this.x += this.vx;
        this.y += this.vy;

        // Boundary Bounce
        if (this.x < 0 || this.x > width) this.vx *= -1;
        if (this.y < 0 || this.y > height) this.vy *= -1;

        // Decay energy
        this.energy *= 0.95;
    }

    draw() {
        // Core
        ctx.beginPath();
        const alpha = 0.3 + this.energy * 0.7;
        ctx.fillStyle = this.energy > 0.1
            ? `rgba(${COLOR_PURPLE}, ${alpha})`
            : `rgba(255, 255, 255, 0.2)`;

        ctx.arc(this.x, this.y, this.radius + this.energy * 5, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Initialize "Brain" Structure
const nodes = [];

// 1. Input Layer (Left - Fixed anchors)
for (let i = 0; i < 5; i++) {
    nodes.push(new Node(50, (height / 6) * (i + 1), true));
}

// 2. Hidden Layers (Floating Physics Particles)
for (let i = 0; i < NODE_COUNT; i++) {
    nodes.push(new Node(width / 2 + (Math.random() - 0.5) * 300, height / 2 + (Math.random() - 0.5) * 300));
}

// 3. Output Layer (Right - Fixed anchors)
for (let i = 0; i < 3; i++) {
    nodes.push(new Node(width - 50, (height / 4) * (i + 1), true));
}

// -- Pulse System (Data Packets) --
const pulses = [];

function spawnPulse() {
    // Start a pulse from a random input node
    const startNode = nodes[Math.floor(Math.random() * 5)]; // First 5 are inputs
    startNode.energy = 1.0;
    pulses.push({
        x: startNode.x,
        y: startNode.y,
        tx: width / 2, // Target general direction
        ty: height / 2,
        life: 1.0,
        path: [] // Could be used for history
    });
}
setInterval(spawnPulse, 200); // 5 data packets per second

// -- Simulation State --
let FUSION_MODE = false;
let INFERENCE_MODE = false;
let ticks = 0;

// -- Simulation Loop --
function update() {
    ticks++;

    // Cycle: 0-300 (Chaos) -> 300+ (Fusion + Inference)
    // The user wants it to "fuse to a smaller grid... and keep going"
    if (ticks > 300) {
        FUSION_MODE = true;
        INFERENCE_MODE = true;
    }

    // 1. Force Directed / Fusion Logic
    // FUSION TARGETS (The ChaosNet Grid)
    if (FUSION_MODE) {
        let gridIndex = 0;
        const cols = 8;
        const gap = 40;
        // Center the grid
        const startX = width / 2 - (cols * gap) / 2 + 20;
        const startY = height / 2 - (5 * gap) / 2;

        nodes.forEach((n, i) => {
            if (!n.fixed) {
                // Calculate target position in grid
                const tx = startX + (gridIndex % cols) * gap;
                const ty = startY + Math.floor(gridIndex / cols) * gap;

                // Strong Pull to Target
                const dx = tx - n.x;
                const dy = ty - n.y;

                // Spring physics to target
                n.vx += dx * 0.05;
                n.vy += dy * 0.05;
                n.vx *= 0.8; // Heavy damping for snapping
                n.vy *= 0.8;

                // Add a tiny "breathing" motion so it's not dead static
                n.x += Math.sin(ticks * 0.05 + i) * 0.2;
                n.y += Math.cos(ticks * 0.05 + i) * 0.2;

                gridIndex++;
            }
        });
    } else {
        // Chaos Forces
        for (let i = 0; i < nodes.length; i++) {
            const n1 = nodes[i];
            for (let j = i + 1; j < nodes.length; j++) {
                const n2 = nodes[j];
                const dx = n2.x - n1.x;
                const dy = n2.y - n1.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < CONNECTION_DIST) {
                    const force = (dist - CONNECTION_DIST) * SPRING_STRENGTH;
                    const fx = (dx / dist) * force;
                    const fy = (dy / dist) * force;

                    if (!n1.fixed) { n1.vx += fx / n1.mass; n1.vy += fy / n1.mass; }
                    if (!n2.fixed) { n2.vx -= fx / n2.mass; n2.vy -= fy / n2.mass; }
                }
            }
        }
    }

    // 2. Update Nodes
    nodes.forEach(n => n.update());

    // 3. Update Pulses (Data Flow)
    // In Inference Mode, pulses travel Left -> Right through grid
    for (let i = pulses.length - 1; i >= 0; i--) {
        const p = pulses[i];

        if (INFERENCE_MODE) {
            p.x += 5; // Fast inference speed
            // Snapping Y to nearest grid row for "Digital" look
            p.y += Math.sin(p.x * 0.1) * 2;
        } else {
            p.x += 8;
            p.y += Math.sin(p.x * 0.02) * 2;
        }

        p.life -= 0.01;

        // Find nearest node to energize
        nodes.forEach(n => {
            const dx = n.x - p.x;
            const dy = n.y - p.y;
            if (dx * dx + dy * dy < 500) { // Closer threshold
                n.energy = Math.min(n.energy + 0.3, 1.0);
            }
        });

        if (p.life <= 0 || p.x > width) pulses.splice(i, 1);
    }
}

function draw() {
    ctx.clearRect(0, 0, width, height);

    // 1. Draw Connections
    ctx.beginPath();

    // In Fusion Mode, draw strict horizontal/vertical lines for "Digital" look
    if (FUSION_MODE) {
        // Draw latent connections between grid neighbors would be complex to map back
        // So we rely on the distance metric which naturally forms the grid lines 
        // due to the positioning.
    }

    for (let i = 0; i < nodes.length; i++) {
        const n1 = nodes[i];
        // Optimization: Only check neighbors in loop
        for (let j = i + 1; j < nodes.length; j++) {
            const n2 = nodes[j];
            const dist = Math.hypot(n2.x - n1.x, n2.y - n1.y);

            // In Fusion mode, we only want neat connections (grid neighbors)
            // Grid gap is 40. Diagonals are ~56.
            const threshold = FUSION_MODE ? 50 : CONNECTION_DIST;

            if (dist < threshold) {
                const alpha = (1 - dist / threshold) * 0.4;
                if (n1.energy > 0.1 || n2.energy > 0.1) {
                    ctx.strokeStyle = `rgba(${COLOR_PURPLE}, ${alpha * 2})`;
                    ctx.lineWidth = FUSION_MODE ? 1.5 : 1.5;
                } else {
                    ctx.strokeStyle = `rgba(${COLOR_ACCENT}, ${alpha * (FUSION_MODE ? 0.3 : 0.5)})`;
                    ctx.lineWidth = 0.5;
                }

                ctx.moveTo(n1.x, n1.y);
                ctx.lineTo(n2.x, n2.y);
            }
        }
    }
    ctx.stroke();

    // 2. Draw Nodes
    nodes.forEach(n => {
        // In Fusion Mode, nodes become uniform
        if (FUSION_MODE) {
            n.radius = 3;
            // Decay energy faster in fusion for twinkling effect
            n.energy *= 0.9;
        }
        n.draw();
    });

    // 3. Draw Pulses
    pulses.forEach(p => {
        ctx.beginPath();
        ctx.fillStyle = '#fff';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#fff';
        ctx.arc(p.x, p.y, FUSION_MODE ? 2 : 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
    });
}

function animate() {
    update();
    draw();
    requestAnimationFrame(animate);
}

// =============================================
// AUTO-START ANIMATION ON LOAD
// =============================================
animate();

// =============================================
// START BUTTON (Optional toggle)
// =============================================

const startBtn = document.getElementById('start-btn');
const btnText = startBtn.querySelector('.btn-text');
const btnIcon = startBtn.querySelector('.btn-icon');

// Set initial state to active
btnText.textContent = 'Network Active';
btnIcon.textContent = '⬤';
startBtn.classList.add('active');

startBtn.addEventListener('click', () => {
    // Scroll to demo section
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
});

// =============================================
// VIDEO CONTROLS + AUTOPLAY ENFORCEMENT
// =============================================

document.addEventListener('DOMContentLoaded', () => {
    // FORCE PLAY on Stable ID Video
    const stableVideo = document.getElementById('video-stable');
    if (stableVideo) {
        stableVideo.muted = true; // Ensure muted for autoplay policy
        const playPromise = stableVideo.play();

        if (playPromise !== undefined) {
            playPromise.then(_ => {
                console.log("Autoplay started!");
            }).catch(error => {
                console.log("Autoplay prevented:", error);
                // Fallback: Add controls overlay or user hint if needed
                // But browser policies usually allow muted autoplay.
            });
        }
    }
});

document.querySelectorAll('.play-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const videoId = btn.dataset.video;
        const video = document.getElementById(videoId);
        const overlay = btn.parentElement;

        if (video) {
            video.play();
            overlay.classList.add('hidden');
        }
    });
});

// =============================================
// NAVIGATION
// =============================================

const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
    let current = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        if (window.scrollY >= sectionTop) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});
