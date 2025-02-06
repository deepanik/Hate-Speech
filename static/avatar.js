// Avatar implementation using Three.js
import * as THREE from 'https://cdn.skypack.dev/three@0.132.2';
import { GLTFLoader } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.132.2/examples/jsm/controls/OrbitControls.js';

class Avatar {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.avatars = [];
        this.mixers = [];
        this.currentConversation = [];
        this.init();
    }

    init() {
        // Set up scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf0f0f0);

        // Set up camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.z = 7;
        this.camera.position.y = 2;

        // Set up renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0, 10, 10);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);

        // Add ground plane
        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.MeshStandardMaterial({ 
            color: 0xcccccc,
            roughness: 0.8,
            metalness: 0.2
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -1;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Add controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI / 2;

        // Load avatars
        this.loadAvatars();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize(), false);

        // Start animation loop
        this.animate();
    }

    async loadAvatars() {
        const loader = new GLTFLoader();
        const avatarPositions = [
            { x: -2, y: 0, z: 0 },
            { x: 2, y: 0, z: 0 }
        ];

        const modelUrl = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@dev/examples/models/gltf/RobotExpressive/RobotExpressive.glb';
        
        try {
            for (let i = 0; i < 2; i++) {
                const gltf = await this.loadModel(loader, modelUrl);
                const avatar = gltf.scene;
                avatar.position.set(avatarPositions[i].x, avatarPositions[i].y, avatarPositions[i].z);
                avatar.scale.set(0.8, 0.8, 0.8);
                avatar.castShadow = true;
                avatar.receiveShadow = true;
                
                // Store avatar and its mixer
                this.avatars.push({
                    model: avatar,
                    mixer: new THREE.AnimationMixer(avatar),
                    animations: gltf.animations,
                    currentAction: null
                });
                
                this.scene.add(avatar);
            }
        } catch (error) {
            console.error('Error loading avatars:', error);
        }
    }

    loadModel(loader, url) {
        return new Promise((resolve, reject) => {
            loader.load(url, resolve, undefined, reject);
        });
    }

    playAnimation(avatarIndex, animationName, loop = true) {
        const avatar = this.avatars[avatarIndex];
        if (!avatar) return;

        // Stop current animation
        if (avatar.currentAction) {
            avatar.currentAction.stop();
        }

        // Find and play new animation
        const animation = avatar.animations.find(anim => anim.name.toLowerCase().includes(animationName.toLowerCase()));
        if (animation) {
            const action = avatar.mixer.clipAction(animation);
            action.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce);
            action.clampWhenFinished = !loop;
            action.play();
            avatar.currentAction = action;
        }
    }

    async performConversation(text, analysisResult) {
        const messageQueue = [];
        
        // First avatar introduces the topic
        messageQueue.push({
            avatar: 0,
            text: "Let me analyze this text for you.",
            animation: "Wave"
        });

        // Add analysis result
        let resultMessage = "";
        let emotion = "";
        
        switch(analysisResult) {
            case 'hate_speech':
                resultMessage = "I've detected hate speech in this content. This type of language can be harmful.";
                emotion = "Angry";
                break;
            case 'offensive_language':
                resultMessage = "This content contains offensive language. While not hate speech, it may be inappropriate.";
                emotion = "Sad";
                break;
            case 'neither':
                resultMessage = "This content appears to be safe and appropriate.";
                emotion = "Happy";
                break;
        }

        messageQueue.push({
            avatar: 1,
            text: resultMessage,
            animation: emotion
        });

        // Process the queue
        for (const message of messageQueue) {
            await this.speakAndAnimate(message.avatar, message.text, message.animation);
            await new Promise(resolve => setTimeout(resolve, 500)); // Pause between speakers
        }
    }

    async speakAndAnimate(avatarIndex, text, animation) {
        return new Promise((resolve) => {
            this.playAnimation(avatarIndex, animation);
            
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.onend = () => {
                this.playAnimation(avatarIndex, "Idle");
                resolve();
            };
            
            // Set different voices for different avatars
            const voices = speechSynthesis.getVoices();
            if (voices.length > 1) {
                utterance.voice = voices[avatarIndex % voices.length];
            }
            
            speechSynthesis.speak(utterance);
        });
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update all mixers
        const delta = 0.016;
        this.avatars.forEach(avatar => {
            if (avatar.mixer) {
                avatar.mixer.update(delta);
            }
        });
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

export default Avatar;
